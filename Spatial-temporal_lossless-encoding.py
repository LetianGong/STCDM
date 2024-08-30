# encoding: utf-8

def encode_numbers(numbers):
    """
    Encoding function interface
    :param numbers: A list of integers to be encoded
    :return: Encoded result (a single integer)
    """
    sorted_numbers = sorted(numbers)
    
    # Patch correction for handling zero
    sorted_numbers = [i + 1 for i in sorted_numbers]
    binary_string = ""
    for num in sorted_numbers:
        binary = bin(num - 1)[2:]  # Convert number to binary, removing the '0b' prefix
        binary = binary.zfill(5)  # Pad with zeros on the left to make the length 5
        binary_string += binary
    encoded = int(binary_string, 2)  # Convert the binary string to a decimal number
    return encoded


def decode_numbers(encoded, n):
    binary_string = bin(encoded)[2:].zfill(5 * n)  # Convert the decimal number to a binary string, padding with zeros on the left
    decoded_numbers = []
    for i in range(n):
        binary = binary_string[i * 5: (i + 1) * 5]  # Group every 5 bits
        number = int(binary, 2) + 1  # Convert the binary string to a decimal number and add 1 to get the original number
        decoded_numbers.append(number)

    # Patch correction for handling zero
    decoded_numbers = [i - 1 for i in decoded_numbers]
    return decoded_numbers


# Example usage
numbers = [8, 9, 15, 20, 3]
n = len(numbers)
encoded_number = encode_numbers(numbers)
decoded_numbers = decode_numbers(encoded_number, n)

print("Encoded number:", encoded_number)
print("Decoded numbers:", decoded_numbers)
