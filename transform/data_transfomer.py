

from os import name


GEN_MAP = {'A': '1', 'C': '2', 'G': '3', 'T': '4' }

def transform(input_file, out_file = 'data/transformed_data.txt', split_length = 22):
    with open(input_file, 'r') as input:
        new_lines = []
        lines = input.readlines()
        for line in lines:
            count = 0
            gen_list = ''
            for i in range(len(line.strip())):
                key = line[i]
                gen_list += GEN_MAP.get(key)
                gen_list += ' -1 '

                count += 1
                if count == split_length:
                    count = 0
                    gen_list += '\n'

                

        
            new_line = gen_list + '-2'
           

            new_lines.append(new_line)

        new_lines = '\n'.join(new_lines)


        with open(out_file, 'w') as output:
            output.write(new_lines)


if __name__ == '__main__':            

    input_file = "/home/sy/Desktop/project/GenCovidAnalysis/data/MT745584.txt"

    transform(input_file=input_file)

    print("Data transformed !!!")

#     # fin = open('/home/sy/Desktop/project/GenCovidAnalysis/data/MT745584TKS,CM-SPAM,ERMIER,PM.txt',"r")
    
#     # line = fin.readline().strip().split(" ")

#     # count = 0
#     # for num in line:
#     #     if num != '-1':
#     #         count += 1


#     # print(count)


# from os import name


# GEN_MAP = {'A': '1', 'C': '2', 'G': '3', 'T': '4' }

# def transform(input_file, out_file = 'data/transformed_data.txt'):
#     with open(input_file, 'r') as input:
#         new_lines = []
#         lines = input.readlines()
#         for line in lines:
#             gen_list = []
#             for i in range(len(line.strip())):
#                 key = line[i]
#                 gen_list.append(GEN_MAP.get(key))

        
#             new_line = ' -1 '.join(gen_list)
#             new_line += ' -1 -2'
           

#             new_lines.append(new_line)

#         new_lines = '\n'.join(new_lines)


#         with open(out_file, 'w') as output:
#             output.write(new_lines)


# if __name__ == '__main__':            

#     input_file = "/home/sy/Desktop/project/GenCovidAnalysis/data/MT745584.txt"

#     transform(input_file=input_file)

#     print("Data transformed !!!")

    





    