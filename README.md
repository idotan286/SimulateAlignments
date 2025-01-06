# BetaAlign: a deep learning approach for multiple sequence alignment
## Motivation:
Multiple sequence alignments are extensively used in biology, from phylogenetic reconstruction to structure and function prediction. Here, we suggest an out-of-the-box approach for the inference of multiple sequence alignments, which relies on algorithms developed for processing natural lan-guages. We show that our AI-based methodology can be trained to align sequences by processing alignments that are generated via simulations, and thus different aligners can be easily generated for datasets with specific evolutionary dynamics attributes. We expect that natural-language processing solutions will replace or augment classic solutions for computing alignments, and more generally, challenging inference tasks in phylogenomics. 

## Results:
The multiple sequence alignment (MSA) problem is a fundamental pillar in bioinformatics, compara-tive genomics, and phylogenetics. Here we characterize and improve BetaAlign, the first deep learn-ing aligner, which substantially deviates from conventional algorithms of alignment computation. Be-taAlign draws on natural language processing (NLP) techniques and trains transformers to map a set of unaligned biological sequences to an MSA. We show that our approach is highly accurate, compa-rable and sometimes better than state-of-the-art alignment tools. We characterize the performance of BetaAlign and the effect of various aspects on accuracy; for example, the size of the training data, the effect of different transformer architectures, and the effect of learning on a subspace of indel-model parameters (subspace learning). We also introduce a new technique that leads to improved perfor-mance compared to our previous approach. Our findings further uncover the potential of NLP-based methods for sequence alignment, highlighting that AI-based algorithms can substantially challenge classic approaches in phylogenomics and bioinformatics.

## Datasets:
Datasets used in this work are available on HuggingFace (Wolf et al., 2020) at: https://huggingface.co/dotan1111.



![image](https://github.com/idotan286/SimulateAlignments/blob/main/Figure_1.png)



Example of aligning three sequences with BetaAlign, (a): (Ⅰ) Consider the unaligned sequences “AAG”, “ACGG” and “ACG”; (Ⅱ) The unaligned sequences are concatenat-ed to a single sentence with a special character “|” between each original sequence; (Ⅲ) The trained model processes the single input sentence and generates the single output sentence; (Ⅳ) The processed output is structured such that the first three nucleotides represent the first column, the next three nucleotides represent the second column, and so on; (Ⅴ) The output is converted into an MSA. (b) An illustration of the different input (Ⅰ) and output (Ⅱ) transformation schemes. (c) Example of handling invalid alignments. When aligning the same sequences, BetaAlign first transformer may mis-takenly mutated the character “A” to “G” (Ⅰ); A different transformer resulted in a different output, may generate a shorter sequence in which the last two characters are missing (Ⅱ); The third transformer provided a valid alignment as output and can be used as the output of BetalAlign (Ⅲ).
