import random
new_file = open("0.2noise_train.txt","w+")
with open("train.txt","r") as file:
	for line in file:
		line = line.strip()
		img_path, label = line.split(' ', 1)
		number = random.uniform(0,1)
		new_label = random.randint(0,6)
		if number <= 0.2:
			while(1):
				new_label = random.randint(0,6)
				if new_label != int(label):
					new_file.write(img_path + ' ' + str(new_label) +'\n')
					break
		else:
			new_file.write(img_path + ' ' + str(label) +'\n')