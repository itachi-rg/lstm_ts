from os import listdir

filedir = '/home/rgangaraju/chaos/selectedFiles_rishikesh/files'
files = [f for f in listdir(filedir)]

cpu_arch = []
block_size = []
data_inst = []
bench = []
prog = []
for f in files:
	name = f.split('.')[0]
	splitnames = name.split('-')
	cpu_arch.append(splitnames[-1])
	block_size.append(splitnames[-2])
	data_inst.append(splitnames[-3])
	bench.append(''.join(splitnames[1:-3]))
	prog.append(splitnames[0])

#print(splitnames)
print(set(cpu_arch))
print(set(block_size))
print(set(data_inst))
print(set(bench))
print(set(prog))
#print(bench)
