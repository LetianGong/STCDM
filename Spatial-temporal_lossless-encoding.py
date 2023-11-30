# encoding: utf-8

def encode_numbers(numbers):
    """
    编码函数接口
    :param numbers: 代编码的多个整数的列表
    :return: 编码后的结果（1个整数）
    """
    sorted_numbers = sorted(numbers)
    
    # 针对零时进行的补丁修正
    sorted_numbers = [i + 1 for i in sorted_numbers]
    binary_string = ""
    for num in sorted_numbers:
        binary = bin(num - 1)[2:]  # 将数字转换为二进制形式，去掉前缀 '0b'
        binary = binary.zfill(5)  # 左侧补零，使其长度为 5
        binary_string += binary
    encoded = int(binary_string, 2)  # 将二进制串转换为十进制数
    return encoded


def decode_numbers(encoded, n):
    binary_string = bin(encoded)[2:].zfill(5 * n)  # 将十进制数转换为二进制串，左侧补零
    decoded_numbers = []
    for i in range(n):
        binary = binary_string[i * 5: (i + 1) * 5]  # 每 5 位进行分组
        number = int(binary, 2) + 1  # 将二进制串转换为十进制数，加上 1 得到原来的数字
        decoded_numbers.append(number)

    # 针对零时进行的补丁修正
    decoded_numbers = [i - 1 for i in decoded_numbers]
    return decoded_numbers


# 示例用法
numbers = [8, 9, 15, 20, 3]
n = len(numbers)
encoded_number = encode_numbers(numbers)
decoded_numbers = decode_numbers(encoded_number, n)

print("Encoded number:", encoded_number)
print("Decoded numbers:", decoded_numbers)
