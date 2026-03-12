import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import spoon.Launcher;
import spoon.reflect.CtModel;
import spoon.reflect.code.CtBlock;
import spoon.reflect.code.CtCodeSnippetStatement;
import spoon.reflect.code.CtExpression;
import spoon.reflect.code.CtReturn;
import spoon.reflect.code.CtStatement;
import spoon.reflect.cu.SourcePosition;
import spoon.reflect.declaration.CtConstructor;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.declaration.CtExecutable;
import spoon.reflect.declaration.CtMethod;
import spoon.reflect.declaration.CtParameter;
import spoon.reflect.factory.Factory;
import spoon.reflect.reference.CtTypeReference;
import spoon.reflect.visitor.filter.TypeFilter;

public final class Defects4JAssertionInjector {
    private static final String INTERNAL_RETURN_VALUE = "__nl2postcond_returnValue";
    private static final String ASSERTION_MARKER = "__NL2POSTCOND_ASSERTION_MARKER__";
    private static final Pattern ASSERT_KEYWORD = Pattern.compile("\\bassert\\b");

    private Defects4JAssertionInjector() {}

    public static void main(String[] rawArgs) throws Exception {
        Arguments args = Arguments.parse(rawArgs);
        Result result;
        try {
            result = run(args);
        } catch (Throwable throwable) {
            result = Result.error(throwable);
        }
        writeResult(args.resultFile, result);
        if (!"ok".equals(result.status)) {
            System.exit(1);
        }
    }

    private static Result run(Arguments args) throws Exception {
        String assertionText = Files.readString(args.assertionFile, StandardCharsets.UTF_8).trim();
        Launcher launcher = new Launcher();
        launcher.addInputResource(args.sourceRoot.toString());
        launcher.getEnvironment().setCommentEnabled(true);
        launcher.getEnvironment().setNoClasspath(true);
        launcher.buildModel();

        CtExecutable<?> executable = findTarget(launcher.getModel(), args.relativeFile, args.methodSignature);
        if (executable.getBody() == null) {
            throw new IllegalStateException("Target executable has no body");
        }

        String rewrittenAssertion = buildInstrumentedAssertion(assertionText);
        String executableKind;
        if (executable instanceof CtConstructor<?>) {
            executableKind = "constructor";
            rewriteConstructor((CtConstructor<?>) executable, rewrittenAssertion, launcher.getFactory());
        } else if (executable instanceof CtMethod<?>) {
            CtMethod<?> method = (CtMethod<?>) executable;
            if (isVoidMethod(method)) {
                executableKind = "void_method";
                rewriteVoidMethod(method, rewrittenAssertion, launcher.getFactory());
            } else {
                executableKind = "method";
                rewriteNonVoidMethod(method, rewrittenAssertion, launcher.getFactory());
            }
        } else {
            throw new IllegalStateException(
                "Unsupported executable type: " + executable.getClass().getName()
            );
        }

        launcher.setSourceOutputDirectory(args.outputRoot.toFile());
        launcher.prettyprint();

        Path rewrittenFile = args.outputRoot.resolve(args.relativeFile);
        if (!Files.exists(rewrittenFile)) {
            throw new IllegalStateException("Spoon did not write rewritten file: " + rewrittenFile);
        }

        List<Integer> assertLines = collectAssertMarkerLines(rewrittenFile);
        if (assertLines.isEmpty()) {
            throw new IllegalStateException("Unable to locate injected assertion marker in rewritten file");
        }

        return Result.ok(executableKind, args.relativeFile, assertLines);
    }

    private static CtExecutable<?> findTarget(
        CtModel model,
        String relativeFile,
        String rawSignature
    ) {
        SignatureParts signature = SignatureParts.parse(rawSignature);
        List<CtExecutable<?>> matches = new ArrayList<>();

        for (CtMethod<?> method : model.getElements(new TypeFilter<>(CtMethod.class))) {
            if (!sameRelativeFile(method, relativeFile)) {
                continue;
            }
            if (matchesMethod(method, signature)) {
                matches.add(method);
            }
        }

        for (CtConstructor<?> constructor : model.getElements(new TypeFilter<>(CtConstructor.class))) {
            if (!sameRelativeFile(constructor, relativeFile)) {
                continue;
            }
            if (matchesConstructor(constructor, signature)) {
                matches.add(constructor);
            }
        }

        if (matches.size() != 1) {
            throw new IllegalStateException(
                "Expected exactly one executable for " + rawSignature + " in " + relativeFile
                    + ", found " + matches.size()
            );
        }
        return matches.get(0);
    }

    private static boolean matchesMethod(CtMethod<?> method, SignatureParts signature) {
        if (signature.constructor) {
            return false;
        }
        if (!method.getSimpleName().equals(signature.executableName)) {
            return false;
        }
        return parametersMatch(method, signature.parameterTypes);
    }

    private static boolean matchesConstructor(
        CtConstructor<?> constructor,
        SignatureParts signature
    ) {
        if (!signature.constructor) {
            return false;
        }
        if (constructor.getDeclaringType() == null) {
            return false;
        }
        if (!constructor.getDeclaringType().getSimpleName().equals(signature.executableName)) {
            return false;
        }
        return parametersMatch(constructor, signature.parameterTypes);
    }

    private static boolean parametersMatch(
        CtExecutable<?> executable,
        List<String> expectedParameterTypes
    ) {
        List<CtParameter<?>> parameters = executable.getParameters();
        if (parameters.size() != expectedParameterTypes.size()) {
            return false;
        }

        for (int index = 0; index < parameters.size(); index++) {
            String actual = normalizeType(parameters.get(index).getType());
            String expected = expectedParameterTypes.get(index);
            if (!typeMatches(actual, expected)) {
                return false;
            }
        }
        return true;
    }

    private static boolean typeMatches(String actual, String expected) {
        if (actual.equals(expected)) {
            return true;
        }
        return stripPackages(actual).equals(stripPackages(expected));
    }

    private static boolean sameRelativeFile(CtElement element, String relativeFile) {
        SourcePosition position = element.getPosition();
        if (position == null || !position.isValidPosition() || position.getFile() == null) {
            return false;
        }
        String expected = normalizePath(relativeFile);
        String actual = normalizePath(position.getFile().getPath());
        return actual.endsWith("/" + expected) || actual.equals(expected);
    }

    private static String buildInstrumentedAssertion(String assertionText) {
        String replaced = assertionText.replaceAll("\\breturnValue\\b", INTERNAL_RETURN_VALUE);
        Matcher matcher = ASSERT_KEYWORD.matcher(replaced);
        if (!matcher.find()) {
            throw new IllegalStateException("Assertion text does not contain an assert statement");
        }
        return matcher.replaceFirst("/* " + ASSERTION_MARKER + " */ assert");
    }

    private static void rewriteVoidMethod(
        CtMethod<?> method,
        String assertionText,
        Factory factory
    ) {
        CtBlock<?> originalBody = method.getBody();
        CtBlock<?> newBody = factory.createBlock();
        if (method.isStatic()) {
            newBody.addStatement(createSnippet(factory, "Object " + INTERNAL_RETURN_VALUE + " = null;"));
        } else {
            String typeName = method.getDeclaringType().getReference().getQualifiedName();
            newBody.addStatement(
                createSnippet(factory, typeName + " " + INTERNAL_RETURN_VALUE + " = this;")
            );
        }
        newBody.addStatement(createSnippet(factory, assertionText));
        for (CtStatement statement : originalBody.getStatements()) {
            newBody.addStatement(statement.clone());
        }
        method.setBody((CtBlock) newBody);
    }

    private static void rewriteConstructor(
        CtConstructor<?> constructor,
        String assertionText,
        Factory factory
    ) {
        CtBlock<?> originalBody = constructor.getBody();
        CtBlock<?> newBody = factory.createBlock();
        for (CtStatement statement : originalBody.getStatements()) {
            newBody.addStatement(statement.clone());
        }
        String typeName = constructor.getDeclaringType().getReference().getQualifiedName();
        newBody.addStatement(
            createSnippet(factory, typeName + " " + INTERNAL_RETURN_VALUE + " = this;")
        );
        newBody.addStatement(createSnippet(factory, assertionText));
        constructor.setBody((CtBlock) newBody);
    }

    private static void rewriteNonVoidMethod(
        CtMethod<?> method,
        String assertionText,
        Factory factory
    ) {
        List<CtReturn<?>> returns = new ArrayList<>();
        for (CtReturn<?> ctReturn : method.getBody().getElements(new TypeFilter<>(CtReturn.class))) {
            if (ctReturn.getParent(CtExecutable.class) == method) {
                returns.add(ctReturn);
            }
        }

        if (returns.isEmpty()) {
            throw new IllegalStateException("Non-void method contains no return statements");
        }

        for (CtReturn<?> originalReturn : returns) {
            CtExpression<?> returnedExpression = originalReturn.getReturnedExpression();
            if (returnedExpression == null) {
                throw new IllegalStateException("Encountered return without expression in non-void method");
            }

            CtBlock<?> replacement = factory.createBlock();
            String returnType = method.getType().clone().toString();
            String returnExpression = returnedExpression.clone().toString();
            replacement.addStatement(
                createSnippet(
                    factory,
                    returnType + " " + INTERNAL_RETURN_VALUE + " = " + returnExpression + ";"
                )
            );
            replacement.addStatement(createSnippet(factory, assertionText));
            replacement.addStatement(createSnippet(factory, "return " + INTERNAL_RETURN_VALUE + ";"));

            originalReturn.replace(replacement);
        }
    }

    private static CtCodeSnippetStatement createSnippet(Factory factory, String text) {
        String snippet = text.trim();
        if (snippet.endsWith(";")) {
            snippet = snippet.substring(0, snippet.length() - 1);
        }
        return factory.Code().createCodeSnippetStatement(snippet);
    }

    private static boolean isVoidMethod(CtMethod<?> method) {
        CtTypeReference<?> type = method.getType();
        if (type == null) {
            return false;
        }
        return "void".equals(type.getSimpleName()) || "void".equals(type.toString());
    }

    private static List<Integer> collectAssertMarkerLines(Path rewrittenFile) throws IOException {
        List<String> lines = Files.readAllLines(rewrittenFile, StandardCharsets.UTF_8);
        List<Integer> markerLines = new ArrayList<>();
        for (int index = 0; index < lines.size(); index++) {
            if (lines.get(index).contains(ASSERTION_MARKER)) {
                markerLines.add(index + 1);
            }
        }
        return markerLines;
    }

    private static String normalizePath(String value) {
        return value.replace('\\', '/');
    }

    private static String normalizeType(CtTypeReference<?> typeReference) {
        if (typeReference == null) {
            return "";
        }
        String raw = typeReference.toString();
        return normalizeType(raw);
    }

    private static String normalizeType(String raw) {
        String normalized = raw.trim().replace("...", "[]");
        normalized = normalized.replaceAll("@[A-Za-z0-9_$.]+", "");
        normalized = normalized.replace("final ", "");
        normalized = normalized.replace("? extends ", "?");
        normalized = normalized.replace("? super ", "?");
        normalized = normalized.replaceAll("\\s+", "");
        return normalized;
    }

    private static String stripPackages(String value) {
        String stripped = value;
        stripped = stripped.replaceAll("([A-Za-z_][A-Za-z0-9_$.]*\\.)+([A-Za-z_][A-Za-z0-9_$]*)", "$2");
        return stripped;
    }

    private static void writeResult(Path resultFile, Result result) throws IOException {
        Files.createDirectories(resultFile.getParent());
        Files.writeString(resultFile, result.toJson(), StandardCharsets.UTF_8);
    }

    private static final class Arguments {
        private final Path sourceRoot;
        private final String relativeFile;
        private final String methodSignature;
        private final Path assertionFile;
        private final Path outputRoot;
        private final Path resultFile;

        private Arguments(
            Path sourceRoot,
            String relativeFile,
            String methodSignature,
            Path assertionFile,
            Path outputRoot,
            Path resultFile
        ) {
            this.sourceRoot = sourceRoot;
            this.relativeFile = relativeFile;
            this.methodSignature = methodSignature;
            this.assertionFile = assertionFile;
            this.outputRoot = outputRoot;
            this.resultFile = resultFile;
        }

        private static Arguments parse(String[] rawArgs) {
            if (rawArgs.length != 12) {
                throw new IllegalArgumentException("Expected 12 arguments, got " + rawArgs.length);
            }

            Path sourceRoot = null;
            String relativeFile = null;
            String methodSignature = null;
            Path assertionFile = null;
            Path outputRoot = null;
            Path resultFile = null;

            for (int index = 0; index < rawArgs.length; index += 2) {
                String flag = rawArgs[index];
                String value = rawArgs[index + 1];
                switch (flag) {
                    case "--source-root":
                        sourceRoot = Path.of(value);
                        break;
                    case "--relative-file":
                        relativeFile = value;
                        break;
                    case "--method-signature":
                        methodSignature = value;
                        break;
                    case "--assertion-file":
                        assertionFile = Path.of(value);
                        break;
                    case "--output-root":
                        outputRoot = Path.of(value);
                        break;
                    case "--result-file":
                        resultFile = Path.of(value);
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown argument: " + flag);
                }
            }

            if (sourceRoot == null
                || relativeFile == null
                || methodSignature == null
                || assertionFile == null
                || outputRoot == null
                || resultFile == null) {
                throw new IllegalArgumentException("Missing required arguments");
            }

            return new Arguments(
                sourceRoot,
                relativeFile,
                methodSignature,
                assertionFile,
                outputRoot,
                resultFile
            );
        }
    }

    private static final class SignatureParts {
        private final boolean constructor;
        private final String executableName;
        private final List<String> parameterTypes;

        private SignatureParts(boolean constructor, String executableName, List<String> parameterTypes) {
            this.constructor = constructor;
            this.executableName = executableName;
            this.parameterTypes = parameterTypes;
        }

        private static SignatureParts parse(String rawSignature) {
            int openParen = rawSignature.indexOf('(');
            int closeParen = rawSignature.lastIndexOf(')');
            if (openParen < 0 || closeParen < openParen) {
                throw new IllegalArgumentException("Invalid signature: " + rawSignature);
            }

            String prefix = rawSignature.substring(0, openParen).trim();
            String parameters = rawSignature.substring(openParen + 1, closeParen).trim();
            boolean constructor = !prefix.contains(" ");
            String executableName = constructor
                ? prefix
                : prefix.substring(prefix.lastIndexOf(' ') + 1);

            List<String> parameterTypes = new ArrayList<>();
            if (!parameters.isEmpty()) {
                for (String parameter : splitTopLevel(parameters)) {
                    parameterTypes.add(normalizeType(removeParameterName(parameter)));
                }
            }
            return new SignatureParts(constructor, executableName, parameterTypes);
        }

        private static List<String> splitTopLevel(String parameters) {
            if (parameters.isEmpty()) {
                return Collections.emptyList();
            }
            List<String> values = new ArrayList<>();
            StringBuilder current = new StringBuilder();
            int genericDepth = 0;
            int arrayDepth = 0;
            int parenthesisDepth = 0;
            for (int index = 0; index < parameters.length(); index++) {
                char currentChar = parameters.charAt(index);
                if (currentChar == '<') {
                    genericDepth++;
                } else if (currentChar == '>') {
                    genericDepth--;
                } else if (currentChar == '[') {
                    arrayDepth++;
                } else if (currentChar == ']') {
                    arrayDepth--;
                } else if (currentChar == '(') {
                    parenthesisDepth++;
                } else if (currentChar == ')') {
                    parenthesisDepth--;
                } else if (currentChar == ','
                    && genericDepth == 0
                    && arrayDepth == 0
                    && parenthesisDepth == 0) {
                    values.add(current.toString().trim());
                    current.setLength(0);
                    continue;
                }
                current.append(currentChar);
            }
            if (current.length() > 0) {
                values.add(current.toString().trim());
            }
            return values;
        }

        private static String removeParameterName(String rawParameter) {
            String parameter = rawParameter.trim();
            int lastSpaceIndex = findLastTypeSeparator(parameter);
            if (lastSpaceIndex < 0) {
                return parameter;
            }
            return parameter.substring(0, lastSpaceIndex).trim();
        }

        private static int findLastTypeSeparator(String parameter) {
            int genericDepth = 0;
            int parenthesesDepth = 0;
            for (int index = parameter.length() - 1; index >= 0; index--) {
                char current = parameter.charAt(index);
                if (current == '>') {
                    genericDepth++;
                } else if (current == '<') {
                    genericDepth--;
                } else if (current == ')') {
                    parenthesesDepth++;
                } else if (current == '(') {
                    parenthesesDepth--;
                } else if (Character.isWhitespace(current)
                    && genericDepth == 0
                    && parenthesesDepth == 0) {
                    return index;
                }
            }
            return -1;
        }
    }

    private static final class Result {
        private final String status;
        private final String executableKind;
        private final String relativeFile;
        private final List<Integer> assertLines;
        private final String message;

        private Result(
            String status,
            String executableKind,
            String relativeFile,
            List<Integer> assertLines,
            String message
        ) {
            this.status = status;
            this.executableKind = executableKind;
            this.relativeFile = relativeFile;
            this.assertLines = assertLines;
            this.message = message;
        }

        private static Result ok(
            String executableKind,
            String relativeFile,
            List<Integer> assertLines
        ) {
            return new Result("ok", executableKind, relativeFile, assertLines, "");
        }

        private static Result error(Throwable throwable) {
            return new Result(
                "error",
                "",
                "",
                Collections.emptyList(),
                throwable.getClass().getSimpleName() + ": " + throwable.getMessage()
            );
        }

        private String toJson() {
            StringBuilder builder = new StringBuilder();
            builder.append("{");
            builder.append("\"status\":\"").append(escape(status)).append("\",");
            builder.append("\"executable_kind\":\"").append(escape(executableKind)).append("\",");
            builder.append("\"relative_file\":\"").append(escape(relativeFile)).append("\",");
            builder.append("\"message\":\"").append(escape(message)).append("\",");
            builder.append("\"assert_lines\":[");
            for (int index = 0; index < assertLines.size(); index++) {
                if (index > 0) {
                    builder.append(",");
                }
                builder.append(assertLines.get(index));
            }
            builder.append("]}");
            return builder.toString();
        }

        private static String escape(String value) {
            return value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r");
        }
    }
}
