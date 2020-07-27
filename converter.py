from Bio import SeqIO
link1=input("Enter the path of dataset:")

fasta_sequences = SeqIO.parse(open(link1),'fasta')
link2=input("Enter the path of file to write:")
file=open(link2,'w')


i=0
for fasta in fasta_sequences:

    name, sequence = fasta.id, str(fasta.seq)
    if (name.find('-') != -1):
        file.write(">"+ str(i) + '|0|training\n'  + sequence + '\n')
    else:
        file.write(">"+ str(i) + '|1|training\n'  + sequence + '\n')

    i+=1
file.close()