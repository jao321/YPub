#!/usr/bin/env python3
import os
import time
import sys

if len(sys.argv) < 3:
	print("You must input two repertoires\n")
	exit()

import platform
from alive_progress import alive_bar
import pandas as pd
import numpy as np
from collections import defaultdict
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

def build_kmers_tf_idf(sequence, ksize=3): 
		ngrams = zip(*[sequence[i:] for i in range(ksize)])
		return [''.join(ngram) for ngram in ngrams]

rep_1 = sys.argv[1]
rep_2 = sys.argv[2]

thr = 0.09
sequence_column = "junction".lower()
vcolumn = "v_call".lower()
jcolumn = "j_call".lower()
seqID = "sequence_id".lower()
separator = "\t"
clone_id =  "clone_id".lower()
separator = "\t"
short_output = False

for x in range(1,len(sys.argv)):
	if sys.argv[x].find("--rep_1") != -1:
		rep_1 = sys.argv[x+1]
	elif sys.argv[x].find("--rep_2") != -1:
		rep_2 = sys.argv[x+1]
	elif sys.argv[x].find("--thr") != -1:
		thr = float(sys.argv[x+1])
	elif sys.argv[x].find("--sequence") != -1:
		sequence_column = sys.argv[x+1]
	elif sys.argv[x].find("--v_gene") != -1:
		vcolumn = sys.argv[x+1]
	elif sys.argv[x].find("--j_gene") != -1:
		jcolumn = sys.argv[x+1]
	elif sys.argv[x].find("--seq_id") != -1:
		seqID = sys.argv[x+1]
	elif sys.argv[x].find("--sep") != -1:
		separator = sys.argv[x+1]
	elif sys.argv[x].find("--clone") != -1:
		clone_id = sys.argv[x+1]	
	elif sys.argv[x].find("--short_output") != -1:
		short_output = True
		out_small_name = filename_temp[0]+"_YClon_clonotyped_only_some_columns."+filename_temp[1]
		if sys.argv[x+1] != "--":
			out_small_name = sys.argv[x+1]




###################################################################################################
#								           repertorio 1                                           #
###################################################################################################

f = open(rep_1, 'r')

x = f.readline().strip()
head = x.split(separator)

head = [col_names.lower() for col_names in head]
number_of_columns = len(head)

try:
	seq_id_indx = head.index(seqID.lower())
except:
	print("\nWARNING\nThere is no column named "+seqID+"\n")
	exit()

try:
	junc_indx = head.index(sequence_column.lower())
except:
	print("\nWARNING\nThere is no column named "+sequence_column+"\n")
	exit()

try:
	vGene_indx = head.index(vcolumn.lower())
except:
	print("\nWARNING\nThere is no column named "+vcolumn+"\n")
	exit()

try:
	jGene_indx = head.index(jcolumn.lower())
except:
	print("\nWARNING\nThere is no column named "+jcolumn+"\n")
	exit()

try:
	clone_id_indx = head.index(clone_id.lower())
except:
	print("\nWARNING\nThere is no column named "+clone_id+"\n")
	exit()


colunas = [head[seq_id_indx],head[junc_indx],head[clone_id_indx]]


rep = "1"

i=0
fail = 0
clonotypes = {}
file_size = 0
print("Processing repertoire 1")
for x in f:
	file_size +=1
	x = x.strip()
	data = x.split(separator)
	if len(data)!= number_of_columns:
		continue
	try:
		if len(data[junc_indx]) < 4:
			continue
		cdr3len = str(len(data[junc_indx]))
		vGene = data[vGene_indx].split('*') #include all v gene alleles
		jGene = data[jGene_indx].split('*') #include all v gene alleles
		key = vGene[0]+","+jGene[0]+","+cdr3len
		clonotypes.setdefault(key, [])
		proclonotype = [data[seq_id_indx]+";"+rep,data[junc_indx], data[clone_id_indx]]
		clonotypes[key].append(proclonotype)
	except:
		fail +=1

	
f.close()


###################################################################################################
#								           repertorio 2                                           #
###################################################################################################

f = open(rep_2, 'r')
x = f.readline().strip()
head = x.split(separator)
head = [col_names.lower() for col_names in head]

number_of_columns = len(head)

try:
	seq_id_indx = head.index(seqID.lower())
except:
	print("\nWARNING\nThere is no column named "+seqID+"\n")
	exit()

try:
	junc_indx = head.index(sequence_column.lower())
except:
	print("\nWARNING\nThere is no column named "+sequence_column+"\n")
	exit()

try:
	vGene_indx = head.index(vcolumn.lower())
except:
	print("\nWARNING\nThere is no column named "+vcolumn+"\n")
	exit()

try:
	jGene_indx = head.index(jcolumn.lower())
except:
	print("\nWARNING\nThere is no column named "+jcolumn+"\n")
	exit()

try:
	clone_id_indx = head.index(clone_id.lower())
except:
	print("\nWARNING\nThere is no column named "+clone_id+"\n")
	exit()


colunas = [head[seq_id_indx],head[junc_indx],head[clone_id_indx]]
rep = "2"


print("Processing repertoire 2")
for x in f:
	file_size +=1
	x = x.strip()
	data = x.split(separator)
	if len(data)!= number_of_columns:
		continue
	try:
		if len(data[junc_indx]) < 4:
			continue
		cdr3len = str(len(data[junc_indx]))
		vGene = data[vGene_indx].split('*') #include all v gene alleles
		jGene = data[jGene_indx].split('*') #include all v gene alleles
		key = vGene[0]+","+jGene[0]+","+cdr3len
		clonotypes.setdefault(key, [])
		proclonotype = [data[seq_id_indx]+";"+rep,data[junc_indx],data[clone_id_indx]]
		clonotypes[key].append(proclonotype)
	except:
		fail +=1

	
f.close()

###################################################################################################
#								     find public clones                                           #
###################################################################################################

temp_filename = open("temp.txt", 'w')


count = 0
total_clust = 0
clones_publicos = {}

for key in clonotypes: #each key is the combination of V gene, J gene and the length of cdr3, the values are the sequence ID and cdr3 sequence
	# if bar.current() == 3336:
	#	 print(clonotypes[key])
	if len(clonotypes[key]) > 1:
		pre_clone = pd.DataFrame(clonotypes[key])
		pre_clone.columns = colunas
		unique_cdr3 = pre_clone.drop_duplicates(sequence_column.lower())
	else:
		unique_cdr3 = clonotypes[key]
	if len(unique_cdr3) > 1:
		# pre_clone = pd.DataFrame(clonotypes[key])
		# pre_clone.columns = colunas
		# unique_cdr3 = pre_clone.drop_duplicates(sequence_column)


		# junc_seq = pre_clone[sequence_column]
		seq_id = pre_clone[seqID.lower()]


		junc_seq = unique_cdr3[sequence_column.lower()]
		seq_unique = unique_cdr3[seqID.lower()]


		# vectorizer = TfidfVectorizer(min_df=1, analyzer=build_kmers_tf_idf)
		vectorizer = CountVectorizer(min_df=1, analyzer=build_kmers_tf_idf)
		tf_idf_matrix = vectorizer.fit_transform(junc_seq)	
		dist = 1 - cosine_similarity(tf_idf_matrix)
		
		clusterer = AgglomerativeClustering(distance_threshold = thr, n_clusters= None, affinity="precomputed",linkage='average')
		
		cluster_pre_clone = clusterer.fit(dist)
		clone = cluster_pre_clone.labels_
		# if cluster_pre_clone.labels_.max() != -1:
		maior = count + cluster_pre_clone.labels_.max() + 1
		# print("clone_size: "+str(len(clone)))
		# print("junc_seq: "+str(len(junc_seq)))
		# print(junc_seq.index)
		junc_seq = junc_seq.reset_index(drop=True) 
		# print(junc_seq[0])
		for i in range(0, len(clone)):
			if clone[i] != -1:
				clone_pub = count + int(clone[i]) +1
			else:
				clone_pub = maior + 1
				maior += 1
				total_clust += 1
			
			# print(junc_seq[i])

			all_again = pre_clone.loc[pre_clone[sequence_column.lower()] == junc_seq[i]][seqID.lower()]
			# print(all_again)
			for k in all_again:
				if str(count) not in clones_publicos:
					clones_publicos[str(count)] = []
					clones_publicos[str(count)].append(str(k))
				else:
					clones_publicos[str(count)].append(str(k))
				temp_filename.write(k+","+str(count)+"\n")	
		count = maior
		total_clust += cluster_pre_clone.labels_.max() + 1	
	else:
		count += 1
		total_clust += 1
		pre_clone = pd.DataFrame(clonotypes[key])
		pre_clone.columns = colunas
		seq_id = pre_clone[seqID.lower()]
		for i in range(0, len(clonotypes[key])):
			if str(count) not in clones_publicos:
					clones_publicos[str(count)] = []
					clones_publicos[str(count)].append(str(seq_id[i]))
			else:
				clones_publicos[str(count)].append(str(seq_id[i]))
			temp_filename.write(str(seq_id[i])+","+str(count)+"\n")


publico = {}
for x in clones_publicos:
	a = [k.split(";")[1] for k in clones_publicos[x]]
	if len(set(a)) != 1:
		for k in clones_publicos[x]:
			publico[k] = x


###################################################################################################
#								          output file                                             #
###################################################################################################

# print(publico)

out = open("convergent_clone"+rep_1.split(".")[0]+"_vs_"+rep_2.split(".")[0]+".tsv","w")

f = open(rep_1, 'r')

check = []

x = f.readline().strip()
out.write(x+separator+"from_file"+separator+"convergent_clone\n")
for x in f:
	temp = x.split(separator)
	if temp[seq_id_indx]+";1" in publico:
		if publico[temp[seq_id_indx]+";1"] not in check:
			out.write(x.strip()+separator+rep_1.replace(".tsv","")+separator+publico[temp[seq_id_indx]+";1"]+"\n")
			check.append(publico[temp[seq_id_indx]+";1"])

f = open(rep_2, 'r')

x = f.readline().strip()
out.write(x+"\n")
for x in f:
	temp = x.split(separator)
	if temp[seq_id_indx]+";2" in publico:
		if publico[temp[seq_id_indx]+";2"] not in check:
			out.write(x.strip()+separator+rep_2.replace(".tsv","")+separator+publico[temp[seq_id_indx]+";2"]+"\n")
			check.append(publico[temp[seq_id_indx]+";2"])


os.remove("temp.txt")

