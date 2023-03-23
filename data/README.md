# Dataset processing instructions

CpG Transformer uses 3 input files:

**(1)** `X.npz`. An encoded genome as a dictionary of NumPy arrays. Every key-value pair corresponds to a chromosome, with the key the name of the chromosome (e.g.: `'chr1'`) and the value a 1D NumPy array of encoded sequence (e.g.: `np.array([0,2,2,3,...,1,1,2])`). Sequences are encoded according to:
```
{'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4,
'M': 5, 'R': 6, 'W': 7, 'S': 8, 'Y': 9,
'K': 10, 'V': 11, 'H': 12, 'D': 13,
'B': 14, 'X': 15}
```

**(2)** `y.npz`. A partially observed methylation matrix as a dictionary of NumPy arrays. Every key-value pair corresponds to a chromosome, with the key the name of the chromosome (e.g.: `'chr1'`) and the value a 2D NumPy array corresponding to the methylation matrix for that chromosome. Every methylation matrix is a `# sites * # cells` matrix with every element at row `i` and column `j` denoting the methylation state of CpG site `i` of cell `j`. Methylation states are encoded by `-1 = unknown`, `0 = unmethylated`, `1 = methylated`

**(3)** `pos.npz`. Positions of all input CpG sites as a dictionary of NumPy arrays. Every key-value pair corresponds to a chromosome, with the key the name of the chromosome (e.g.: `'chr1'`) and the value a 1D NumPy array corresponding to the locations of all profiled CpG sites in that chromosome (columns in the second input).

Example:

```python
>>> X['chr1'].shape
(197195432,)
>>> X['chr1']
array([4, 4, 4, ..., 1, 1, 2], dtype=int8)

>>> y['chr1'].shape
(1190072, 20)
>>> y['chr1']
array([[-1,  1, -1, ..., -1,  1, -1],
       [-1,  1, -1, ...,  1, -1, -1],
       [-1,  0, -1, ...,  0, -1, -1],
       ...,
       [-1, -1, -1, ..., -1, -1, -1],
       [-1, -1, -1, ..., -1, -1, -1],
       [-1, -1, -1, ..., -1, -1, -1]], dtype=int8)

>>> pos['chr1'].shape
(1190072,)
>>> pos['chr1']
array([  3000573,   3000725,   3000900, ..., 197194914, 197194986,
       197195054], dtype=int32)
```

# From .tsv files

We provide a simple script `EncodeFromTsv.py` that will convert a .tsv file with columns: `1: chromosome, 2: position on chromosome, 3-...: methylation calls for cells (-1/0/-1)`.

Example:
```
chr1    300356  1       -1      0       -1
chr1    300894  1       0       -1      -1
chr1    301856  -1      -1      0       0
...
chrY    185123  -1      0       -1      0
chrY    185627  0       -1      -1      -1
chrY    185823  -1      -1      0       -1
```

**New:** `EncodeFromTsv.py` has support for tab-separated files with continuous methylation calls via the `--continuous` flag.

# Benchmark datasets instructions

## Ser dataset

#### Genome
[Link](ftp.ensembl.org/pub/release-67/fasta/mus_musculus/dna/)
```bash
mkdir Ser
cd Ser
wget -nH --cut-dirs=7 -r ftp://ftp.ensembl.org/pub/release-67/fasta/mus_musculus/dna/*dna.chromosome*
gunzip *
cat * > genome.fa
rm *dna.chromosome*
```

Use grep to find the order of chromosomes:
```bash
grep ">" genome.fa
```

And encode, using this order as input argument to `EncodeGenome.py`:
```bash
python ../EncodeGenome.py genome.fa X.npz --chroms 10 11 12 13 14 15 16 17 18 19 1 2 3 4 5 6 7 8 9 MT X Y --prepend_chr
```

#### Methylation matrix and positions

[Link](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE56879)

To download:
```bash

count=0
for i in $(seq 55 74)
do
    count=$((count+1))
    wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1370nnn/GSM13705"$i"/suppl/GSM13705"$i"_Ser_"$count".CpG.txt.gz
done

gunzip *CpG.txt.gz
```

Encode all files and combine.
```bash
count=0
for cell in $(ls *CpG.txt)
do
    count=$((count+1))
    python ../EncodeLabelsSmallwood.py $cell X.npz y_"$count".npz pos_"$count".npz --prepend_chr --chroms 10 11 12 13 14 15 16 17 18 19 1 2 3 4 5 6 7 8 9 MT X Y
done

python ../CombineEncodedLabels.py --y_files y_* --pos_files pos_* --y_outFile y.npz --pos_outFile pos.npz
```

(Optionally) remove processing files. Doing this you will only keep the three input files for the model.
```bash
rm *.fa *.txt pos_* y_*
```

Thanks to [yuzhong-deng](https://github.com/yuzhong-deng) for providing a script to download genomic contexts to evaluate predictions on specific subsets of the genome corresponding to a specific annotation.
See [his notebook](https://github.com/gdewael/cpg-transformer/blob/main/data/genomic-contexts/genomic_contexts_data.ipynb).


## 2i dataset

Perform the same steps as with the Ser dataset. Change the downloaded methylation matrix files to the 2i files by doing: `for i in $(seq 35 46)` instead of `for i in $(seq 55 74)`

## HCC dataset

#### Genome
[Link](ftp.ncbi.nlm.nih.gov/genomes/archive/old_genbank/Eukaryotes/vertebrates_mammals/Homo_sapiens/GRCh37/Primary_Assembly/assembled_chromosomes/FASTA/)
```bash
mkdir HCC
cd HCC
wget -nH --cut-dirs=10 -r ftp://ftp.ncbi.nlm.nih.gov/genomes/archive/old_genbank/Eukaryotes/vertebrates_mammals/Homo_sapiens/GRCh37/Primary_Assembly/assembled_chromosomes/FASTA/*fa.gz
gunzip *
cat * > genome.fa
rm *dna.chromosome*
```

Use grep to find the order of chromosomes:
```bash
grep ">" genome.fa
```

And encode, using this order as input argument to `EncodeGenome.py`:
```bash
python ../EncodeGenome.py genome.fa X.npz --chroms 10 11 12 13 14 15 16 17 18 19 1 20 21 22 2 3 4 5 6 7 8 9 X Y --prepend_chr
```

#### Methylation matrix and positions

[Link](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65364)

To download:
```bash

count=0
for i in $(seq 767 2 815)
do
    count=$((count+1))
    formatcount=$(printf "%02d" $count)
    wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1593nnn/GSM1593"$i"/suppl/GSM1593"$i"_Ca_"$formatcount"_RRBS.single.CpG.txt.gz
done

gunzip *CpG.txt.gz
```

Encode all files and combine.
```bash
count=0
for cell in $(ls *CpG.txt)
do
    count=$((count+1))
    python ../EncodeLabelsHCC.py $cell X.npz y_"$count".npz pos_"$count".npz --prepend_chr --chroms 10 11 12 13 14 15 16 17 18 19 1 20 21 22 2 3 4 5 6 7 8 9 X Y 
done

python ../CombineEncodedLabels.py --y_files y_* --pos_files pos_* --y_outFile y.npz --pos_outFile pos.npz
```

(Optionally) remove processing files. Doing this you will only keep the three input files for the model.
```bash
rm *.fa *.txt pos_* y_*
```

## MBL dataset

#### Genome
Same as HCC.

```bash
mkdir MBL
cd MBL
...
```

#### Methylation matrix and positions


[Link](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE125499)

To download:
```bash

for i in $(seq 56 85)
do
    wget -r -nH -np --cut-dirs=4 -e robots=off -A *.bw https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4912nnn/GSM49120"$i"/suppl/
done

mv suppl/*.bw ./
rm -r suppl
```

Convert BigWig to Wig.
```bash
wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64.v385/bigWigToWig

for file in $(ls *.bw)
do
    ./bigWigToWig $file ${file:0:-2}wig
done
```

Encode all files and combine.
```bash
count=0
for cell in $(ls *.wig)
do
    count=$((count+1))
    python ../EncodeLabelsMBL.py $cell X.npz y_"$count".npz pos_"$count".npz --prepend_chr --chroms 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22
done

python ../CombineEncodedLabels.py --y_files y_* --pos_files pos_* --y_outFile y.npz --pos_outFile pos.npz
```

(Optionally) remove processing files. Doing this you will only keep the three input files for the model.
```bash
rm *.fa *.txt pos_* y_*
```


## Hemato dataset

#### Genome
[Link](ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/)
```bash
mkdir Hemato
cd Hemato
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/GCA_000001405.15_GRCh38_genomic.fna.gz
gunzip GCA_000001405.15_GRCh38_genomic.fna.gz
mv GCA_000001405.15_GRCh38_genomic.fna genome.fa
../fasta_manipulation/FastaToTbl genome.fa | grep -E 'CM|mito' | ../fasta_manipulation/TblToFasta > genome2.fa
```

Use grep to find the order of chromosomes:
```bash
grep ">" genome2.fa
```

And encode, using this order as input argument to `EncodeGenome.py`:
```bash
python ../EncodeGenome.py genome2.fa X.npz --chroms 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X Y M --prepend_chr
```

#### Methylation matrix and positions

[Link](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87197)

To download:
```bash
for i in $(seq 443 463; seq 497 515; seq 549 570; seq 625 642; seq 815 838; seq 977 994)
do
    wget -nH --cut-dirs=5 -r ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2324nnn/GSM2324"$i"/suppl/*
done

gunzip *txt.gz
```

Encode all files and combine.
```bash
count=0
for cell in $(ls GSM*.txt)
do
    count=$((count+1))
    python ../EncodeLabelsFarlik.py $cell X.npz y_"$count".npz pos_"$count".npz --prepend_chr --chroms 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X Y M 
done

python ../CombineEncodedLabels.py --y_files y_* --pos_files pos_* --y_outFile y.npz --pos_outFile pos.npz
```

(Optionally) remove processing files. Doing this you will only keep the three input files for the model.
```bash
rm *.fa *.txt pos_* y_*
```


