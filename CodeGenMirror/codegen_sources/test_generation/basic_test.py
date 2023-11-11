from select_java_inputs import * 

java_processor = JavaProcessor(TREE_SITTER_ROOT)


def is_simple_standalone_func(func):
    global java_processor
    try:
        args = java_processor.extract_arguments(func)
        return_type = extract_return_type_java(func)
        print(f"func: {func} args: {args} return: {return_type}")
        if all(
            [
                arg.replace("final ", "").replace(" ", "")
                in java_supported_types | {"None"}
                for arg in args[0]
            ]
        ) and return_type in java_supported_types | {"void"}:
            if (
                return_type == "void"
                and not any(
                    [
                        "[]" in arg.replace(" ", "") or "List" in arg or "Array" in arg
                        for arg in args[0]
                    ]
                )
                or java_processor.get_function_name(func).strip() == "main"
            ):
                return False
            print("compilation: ", get_java_compilation_errors(
                java_processor.detokenize_code(func), timeout=120))
            if (
                get_java_compilation_errors(
                    java_processor.detokenize_code(func), timeout=120
                )
                == "success"
            ):
                return True
        return False
    except ValueError:
        return False
    except IndexError:
        return False

f = 'public static int add ( int x , int y ) { return x + y ; }'


print(f"is {f} simple standalone: {is_simple_standalone_func(f)}")
