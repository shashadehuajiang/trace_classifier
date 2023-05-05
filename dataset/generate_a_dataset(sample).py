import random
import json

vector1 = [0,1]
vector2 = [1,1]
vector3 = [2,1]
vector4 = [3,-1]
vector5 = [4,-1]
vector6 = [5,-2]

flow1 = [vector1,vector2,vector3]
flow2 = [vector1,vector2,vector3]
flow3 = [vector1,vector3]
flow4 = [vector1,vector2]

flow5 = [vector1,vector4,vector5]
flow6 = [vector1,vector2,vector4]
flow7 = [vector1,vector6]
flow8 = [vector1,vector6,vector2]


sample_list = []
for i in range(1000):
    label = 0
    trace = random.sample([flow1,flow2,flow3,flow4], random.randint(1,3)) 
    one_sample = [trace, label]
    sample_list.append(one_sample)

    label = 1
    trace = random.sample([flow5,flow6,flow7,flow8], random.randint(1,3)) 
    one_sample = [trace, label]
    sample_list.append(one_sample)


# save json file
dataset = sample_list
label2key = [0, 1] # label list


print('save dataset...')
content = [dataset,label2key]
filename = "test.json"
with open(filename, 'w') as file_obj:
    json.dump(content, file_obj)

print("one_sample:", sample_list[0])