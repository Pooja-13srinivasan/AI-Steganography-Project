import cv2

def extract_lsb_message(image_path):

    img = cv2.imread(image_path)

    binary_data = ""

    for row in img:

        for pixel in row:

            for channel in pixel:

                binary_data += str(channel & 1)

    chars = [binary_data[i:i+8] for i in range(0,len(binary_data),8)]

    message = ""

    for char in chars:

        ascii_char = chr(int(char,2))

        if ascii_char == "#":
            break

        message += ascii_char

    return message