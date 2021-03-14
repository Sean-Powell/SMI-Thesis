LENGTH = 1000000

f = open("C:/Users/seanp/Desktop/Thesis/Results/GA Functions 6000/output.txt")

current_line = 0
current_file = None
current_file_index = 0


def create_file():
    global current_file_index
    temp_f = open("C:/Users/seanp/Desktop/Thesis/Results/GA Functions 6000/output_" + str(current_file_index) + ".txt", "w")
    current_file_index += 1
    return temp_f


for l in f:
    if current_file is None:
        current_file = create_file()
    if current_line > LENGTH:
        current_file.close()
        current_file = create_file()
        current_line = 0
    current_file.write(l)
    current_file.write("\n")
    current_line += 1




