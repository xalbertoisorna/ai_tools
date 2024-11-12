import os
import re


def translate_int8_to_int16_mlir(mlir_code):
    """
    Translates MLIR code from using int8 quantization to int16 quantization, removing any zero-point specification.

    Args:
    mlir_code (str): The original MLIR code string with int8 quantization.

    Returns:
    str: Translated MLIR code with int16 quantization.
    """
    # Step 1: Replace int8 quantization with int16 quantization in tensor types
    mlir_code = re.sub(
        r"!quant\.uniform<i8:f32, ([^>]+)>", r"!quant.uniform<i16:f32, \1>", mlir_code
    )

    # Step 2: Remove any zero-point by eliminating it from the parameter list
    mlir_code = re.sub(
        r"!quant\.uniform<i16:f32, ([^,:]+):[^>]+>",
        r"!quant.uniform<i16:f32, \1>",
        mlir_code,
    )

    return mlir_code


def process_mlir_files_in_directory():
    """
    Processes all .mlir files in the current directory by translating int8 quantization to int16,
    and saving the output to a new file with the _int16.mlir suffix.
    """
    for filename in os.listdir("."):
        if not filename.endswith(".mlir") or filename.endswith("_int16.mlir"):
            continue
        with open(filename, "r") as file:
            mlir_code = file.read()

        # Translate the MLIR code
        translated_mlir_code = translate_int8_to_int16_mlir(mlir_code)

        # Save the translated code to a new file
        new_filename = f"{os.path.splitext(filename)[0]}_int16.mlir"
        with open(new_filename, "w") as new_file:
            new_file.write(translated_mlir_code)
        print(f"Processed {filename} -> {new_filename}")


# Execute the script
if __name__ == "__main__":
    process_mlir_files_in_directory()
