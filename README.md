# nlp
### NLP BIOBERT
### Fall 2021, CU Boulder
###### Professor James H. Martin
###### Student: Sushma Akoju

Note: This is part of a class work and the datasets are not supposed to be publicly shared. The software libraries used in this work need to be a specific version from the year this notebook was run. Please note that the reason this work was shared much later after this work was done. This is because newer versions of software libraries came up and massive changes in how NLP models were used for this task have also evolved. Therefore this notebook may not run or work in exact same way due to mismatch of versions. Appreciate anyone who have been contacting me to share the datasets. 

Thanks to everyone who found this notebook was helpful for their learning purposes and appreciate you reaching out to me to mention the same.

## Code of conduct
Resources were used for specific literature/code, it is provided in the respective implementation file. The code provided here is implicitly expected not to be replicated for homeworks, assignments or any other programming work. It is welcome to take inspiration, but it is implicitly expected to cite this resource if used for learning, inspiration esp. for any homeworks/projects purposes to respect the policies of your respective institution and the academic integrity for your class. Please refer <a href="https://github.com/sushmaakoju/nlp-bio-bert/blob/main/CODE_OF_CONDUCT.md"> code of conduct</a>.

## Goal
This is purely exploratory: the goal is to learn and practice more.

### Focussed only on my learning and notes from the analysis:
The task was mainly focussed on data that has BIO tags over biomedical data. 
The data with gene, proteins and biochemical interactions has its own Finite state automata and grammar.
Thus this requires state-of-the-art NLP models that are specific to the biomedical data.
<a href="https://bair.berkeley.edu/blog/2019/11/04/proteins/">Can We Learn the Language of Proteins?</a>

### Task:

Submit a report of what you did, your python code, and test results.  I will post a test set before the due date.  Run the test data through your system and submit the output.  The output format of your system should be the same as the input training data: one token per line, with a tab separating the token from its tag, and a blank line separating sentences.

#### Analysis of Dataset:
The dataset seems like a text from a medical journal consisting of articles on biomedical and/or clinical topics with sufficient reference to gene, proteins, protein interactions and biochemical interactions. The data represents I,O,B tags for each word for each sentence in the dataset. There are about 13,795 sentences. The vocabulary is different from regular English texts. The task in homework suggests to detect “Gene” entities which could appear in patterns such as “BII”, “BIO”, or even regex such as “BI[I]*[O]*”.

The dataset and vocabulary likewise, underpinned by specific nomenclature, the scientific names of proteins, DNA structures, interactions of proteins, which indicate sufficient intersection of biomedical literature and the domain knowledge required to understand word embeddings, for example. The dataset brings forth interesting patterns such as having simple, english vocabulary appearing almost randomly placed next to biomedical literature. Despite such intricate patterns of word combinations, subword information, the information in the dataset seems to provide minimalistic features: word, tag combinations for each sentence in the dataset. The subword information seems to be key information required for this dataset.

### Define the problem: 
Given a word in a sentence, identify the one tag that can most likely represent the entity i.e. from the tags (B,I,O).

Entity Sequence required for this problem:
Understanding if a gene exists from a sequence of tags. Gene is likely defined by “BII” sequence of tags or “BI[I]*[O]*” irrespective of previous tags.

#### Analysis about dataset:
Dataset is far apart from regular vocabulary such as the ones from other non-domain specific tasks such as POS tagging over BERT models. There are several rare words. There are rare words. 
A lot of medical terms, bio medical terminology from this dataset are rare with frequencies <10.

About the dataset:

Total number of sentences: 13795
Total number of words/tokens in dataset: 308229
Max number of words in a sentence: ~102
Vocabulary size: 27282

Total number of B tags: 13304
Total number of I tags: 19527
Total number of O tags: 276009

O tags clearly are more highly common than B and I tags.
The max number of B tags a word can have is 265 compared to the max number of O tags. Similarly, max number I tags are 4393 which is about 25% of O tags.

<img width="500" height="300" src="https://user-images.githubusercontent.com/8979477/143825097-ec020cfc-ffc9-42d2-8637-85bf060a99c3.png">


<img width="500" height="300" src="https://user-images.githubusercontent.com/8979477/143825095-ee320a4f-6196-4e2d-8362-05744373f96b.png">
The most common top 10 words are: “.”, “the”, “of”, “-”, (',','and','in','a','(', 'to'.

<img width="400" height="25" src="https://user-images.githubusercontent.com/8979477/143825094-9da9fdac-8d05-4ea6-be35-58ae64c3e510.png">

The least common top 10 words are: 'K713','hypercholesterolemic','lutein','P69','conference','Talk','Tele','cruciform','TE105'

<img width="400" height="25" src="https://user-images.githubusercontent.com/8979477/143825093-67cc6edb-3d42-4bd4-8dd2-788667172adf.png">

Most importantly, most of the tokens that are from biomedical /bioinformatics terminology itself are very less frequent or rare words.

The facts for the problem from the dataset:
Several words in the dataset have any of the three tags from B,I,O.
Given a sequence of tags: “t1,t2,t3” Each tag t_i is more likely to occur along with the previous tag t_i-1 and next the tag t_i+1 depending on the Markov property that applies conveniently to this tag sequence problem.
Each word can have only one tag.
Some words are labelled for one of 3 tags, for different sentences.
Some tag sequences and each of their frequencies:

The two tag sequences and their frequencies:
<img width="400" height="25" src="https://user-images.githubusercontent.com/8979477/143825090-c0860bf5-3d5a-4dd9-b449-67f99c9a849a.png">

The three tag sequences and each of their frequencies:

<img width="120" height="120" src="https://user-images.githubusercontent.com/8979477/143825092-a34adbdd-e2f4-4c75-89f4-f1c151514f1c.png">

The tag frequencies suggest that OOO is most frequent and OOB, OBI, IOO, OBO, BII are most frequent.

#### HMM with Viterbi Approach:

First I attempted to implement the Hidden Markov Model with Viterbi for the task. Given, transition probability matrix, emission probabilities, and initial probabilities of states, tags and words, it appears easier to calculate the results and predicted tags for each word. The HMM model with Viterbi, the approach as follows:

Defining input probabilities:

States: States in this case are tag sequences. 
Define 2 tag sequence for Bi-gram case: The given tags : B,I,O and have 2^3 = 8 arrangements including repetitions
Define 3 tag sequence for tri-gram case: The given tags : B,I,O and have 3^3 = 27 arrangements including repetitions. 
Solving for the most common subsequence: This can be done by making all tag sequences into a single string of all patterns. The resulting super sequence of all tags is as follows:“OOOOOOOOOOOOOBIIIOBIIOOOOOOOOOOOOOOOOBIIIOOOOOOOOBIII…..”
Algorithmically and programmatically, the sequence counts for each state i.e tag sequence such as “IB”, “BII”, “BIO”  can be generated using Counter data structure (or a suffix algorithm or other advanced (string, index) sequence extraction algorithms).
There are about 92 sequences of “BII” in the super sequence string.
The transition probability for “I” given B will be: count(tag == B) / count(tag == “BI”)
Total single tag counts i.e. count (B), count(I), count(O) = 'B': 13363, 'I': 19779, 'O': 276565
All possible two-tag sequences:
'BB', 'BI', 'BO', 'IB', 'II', 'IO', 'OB', 'OI', 'OO'
Counts of 2-tag sequences: 
The two tag sequences and their frequencies:

<img width="400" height="25" src="https://user-images.githubusercontent.com/8979477/143825086-009da641-9901-4e17-a2d5-e4923d2f654c.png">

All possible 3-tag sequences: IBO BII IIB BOB IOI OII OBI BIO IBI OBO BOI BIB IOB IOO OIB OBB OIO BBO OOO BBI BBB BOO OOB IBB IIO OOI III


Counts if 3-tag sequences
The three tag sequences and each of their frequencies:

<img width="150" height="250" src="https://user-images.githubusercontent.com/8979477/143825084-2fbb5c37-2282-420f-bc94-6e1a738ade2b.png">

Transition probabilities:
The probability of going from one state to another assumes Markov independence assumption.
Bi-gram case: product p(t_1 | t_i-1 )

Tri-gram case: product (t_1 | t_i-1, t_i-2 )

<img width="250" height="175" src="https://user-images.githubusercontent.com/8979477/143825080-da3523a7-10dd-48da-b0e6-4fd3c2834599.png">


Emission probabilities:
The probability of emitting a word given a tag
For 2-tag sequences:

<img width="300" height="200" src="https://user-images.githubusercontent.com/8979477/143825078-dd757846-309d-4a3a-b80f-a3fc1b0512c6.png">

For 3-tag sequences: 

<img width="350" height="250" src="https://user-images.githubusercontent.com/8979477/143825076-428c2722-e5e2-4b55-a91e-3d7d68cd0d86.png">

Building a word count for each word and tag sequence (which can be used to build trellis for dynamic programming:

<img width="700" height="175" src="https://user-images.githubusercontent.com/8979477/143825074-be15ca85-c462-44a8-8e78-b62c2913f4ba.png">

Results:
Finally after 80/20 split over data, after training HMM with Viterbi, as expected, almost always O tag was predicted while this would increase False Positives. I saw a 0.327 f1-score.



The tag-wise confusion matrix, F1-score, precision and recalls results for HMM Viterbi are as follows:
<img width="300" height="100" src="https://user-images.githubusercontent.com/8979477/143825070-502d75b1-c10e-4829-8e7d-97de74885920.png"><img width="450" height="50" src="https://user-images.githubusercontent.com/8979477/143825072-c0a701a0-a1ce-42e7-a8c2-a22b9f0dbfb2.png">

The f-score for I and B tags is better however it seems pretty good for O tag since O is the most frequent tag.

Summary of results:
From above results, f1-score, it is not recommended to continue using HMM and Viterbi approach for Named Entity Recognition, since the pitfall of HMM model is that it is not flexible enough to unknown words as well as any new vocabulary. It is possible to consider the previous word is a B tag, then next tag is an O tag, then the likelihood of having an I tag or O tag more than a B tag. However this consideration is not sufficient to generalize and improve HMM with Viterbi. 

We could use additional features such as the first letter is a capital letter, all letters are capital letters, the previous word is a hyphen, or next word is a number, previous word + next word is an alphanumeric, all of which can act as better features for working with HMM. The idea is frequencies for each of these new features and their likelihood of being assigned a tag under a 5-gram approach; without a pretrained model, the HMM model might survive for small problems but for small, limited real world applications.

### IB Tag sequence without an O tag with ”en_core_web_md”:

Given that Data analysis and statistical analysis of data sets surrounding words, tags and their corresponding frequencies from the above sections, I found that it is perhaps better to ignore O tag. The tags that are considered are I, B tags only. Once after prediction of I, B tags is complete, we could assume all of the remaining untagged words are O tags.

So in this approach, I trained the en_core_web_md model using the SpACY library and extracted entities (with start and end ids for each word in a sentence) and processed it through a minimalistic SpACY approach.

Evaluation
The validation during model training task, the precision, f1-score and recall scores are as follows for the I, B tags for NER task:

<img width="400" height="300" src="https://user-images.githubusercontent.com/8979477/143825067-8083283d-74ea-4d7e-baad-c51795c4e284.png"><img width="400" height="300" src="https://user-images.githubusercontent.com/8979477/143825069-befecee1-be2f-4f52-ab59-07e8d767f572.png">

The precision and recall along with the f1-scores for test data are worse as expected. This evaluation was conducted based on the evaluation script provided.

<img width="550" height="50" src="https://user-images.githubusercontent.com/8979477/143825065-82eb0251-9a4e-400f-8dfb-6ba7dd5bf564.png">

And finally on calculating tag-wise confusion matrix, f1-score, precision and recall, the results are as follows:
<img width="250" height="100" src="https://user-images.githubusercontent.com/8979477/143825066-a0a2f154-c1dc-43ee-8fc7-96d21d59cf7e.png">

The f1-score is very low and the true positives are significantly low and almost zero. This model and approach is not suitable and hence should not be considered.

Summary of results: an O tag is very important in recognizing frequent sequence patterns towards training the model towards the NER task for this dataset with IOB tags.

### BIO tag sequence with ”en_core_web_md”:

Given that Data analysis and statistical analysis of data sets surrounding words, tags and their corresponding frequencies and based on results from IB tag sequence from above section, the next natural assumption is to include O tag nevertheless.

So in this approach, I just trained the en_core_web_md model using the SpACY library and extracted entities (with start and end ids for each word in a sentence) and processed it through a minimalistic SpACY approach.

Evaluation
The validation results during training phase and the f1-score, recall and precision scores are as follows:


<img width="400" height="250" src="https://user-images.githubusercontent.com/8979477/143825060-d58fc7e7-10a1-418d-9ac2-81895b72dc7c.png">

<img width="400" height="250" src="https://user-images.githubusercontent.com/8979477/143825063-3a951041-4fb9-48c0-98b0-77a96bf9d457.png">

The precision and recall along with the f1-scores for test data are worse as expected. This evaluation was conducted based on the evaluation script provided.


<img width="500" height="50" src="https://user-images.githubusercontent.com/8979477/143825058-af882b4c-a162-4348-948a-dbc0c0a408a6.png">

Additionally, upon tag-wise confusion matrix, f1-score, precision and recall, which are fall further (even lesser than previous case):

<img width="300" height="80" src="https://user-images.githubusercontent.com/8979477/143825057-39a39b09-b5e2-4d74-95bc-5b81a8608aef.png">

Summary of results: this model is not suitable for Named Entity tagging for Bioinformatics data. The vocabulary consists of too many rare words and clearly the model is not adaptable to unknown or rare words that are significantly different from words that do exist in vocabulary. Due to the fact that there are 27000+ words in this dataset, most of them with very different subwords(such as names originating from chemistry, biology), it could be helpful to use a more domain specific trained model. 

### BioBERT 
After a little bit of research on Google scholar and NIH, BioBERT is a fast, and vastly used model in Bioinformatics. BioBERT used BioInformatics data and is different from BERT. The difference lies in first considering subwords and finally enhancing the word2Vector embeddings for the bioinformatics data. This seemed a reasonable direction to explore, without hoping much too soon.

A special case to consider, for example, the least common words in dataset provided for this homework, 'K713','hypercholesterolemic','lutein','P69','conference','Talk','Tele','cruciform','TE105'. They are not only least common, they need special domain specific knowledge for subword tokenization, which is very different as well as difficult from that of other common English word tokenizations. Thus BioBERT makes for a special case and is more reasonable to explore. This also complies with the fact that domain specific expertise adds additional information required to understand the Named Entity tags and vice versa. The reverse case is : to represent knowledge and “reason” from a medical journal text corpus, Named entity recognition is required for identifying domain specific instances. To rephrase the reverse case, for domain specific knowledge mining, we need Named Entity recognition as a prerequisite. 

However, the BioBERT or generally BERT models do consider POS tags along with entity tags. So few tasks for conducting Named Entity recognition and training a model for this task, following were the steps:
As is the case, firstly I generated POS tags using SpACY library for each of the words for each sentence provided in the dataset. So the POS tag is a new added feature to this dataset.
Secondly, train the model evaluate the model.

<img width="500" height="250" src="https://user-images.githubusercontent.com/8979477/143825055-d6c2806a-c76e-4c41-98ce-62b960a21673.png">
<!-- <img width="500" height="300" src="eval_biobert](https://user-images.githubusercontent.com/8979477/143825055-d6c2806a-c76e-4c41-98ce-62b960a21673.png)
 -->
The approach is simple and it seemed at first to reasonably work well.
The validation during training as in other cases, gave 84%
Upon tag-wise results with confusion matrix, f1-score, precision and recall are as follows, which are also very poor.

<img width="400" height="70" src="https://user-images.githubusercontent.com/8979477/143825052-dc1b7201-e798-4e9f-a489-2ccc8ae7628e.png">

Summary of results: POS tags are important information to identifying Named Entity tags and yet having additional features would contribute better. Additionally when training smaller models over BERT/BERT extended models, the need for more data seems one expectation. There are additional cases upon researching a little further. The structure considerations: in most of the predictions from above three tasks, some proteins are almost always recognized correctly due to the co-occurrence with certain words. They are more likely to occur with other rare words. However, the language of proteins in chemistry and biological data seems to contribute more intricately towards co-occurrence. That is to say the representation style, structure, format, sequence of capital letters with numbers to showcase a chemical compound or a double helix structure, symbolic representation contribute significantly towards linguistic tasks. For example a structure of gene, might naturally suggests that TE105 is a representation of 105-nt TE
(TE105).[https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=1040&context=plantpath_pubs ]. So if a word occurred such as “105-nt TE
(TE105)”, a pretrained model over biomedical/bioinformatics text corpus might be more likely to recognize this word and identify subwords for text representation. Navigating a little outside scope of this topic, the article [https://bair.berkeley.edu/blog/2019/11/04/proteins/] suggests that understanding protein structures is considered similar to a Linguistics task. This is sufficient evidence to infer that any domain specific knowledge representation requires significant understanding of underlying structures whether the data is a biomedical textual data or as complex as a domain specific representation such as that of proteins, DNA bindings in a freeform text.

### Other methods attempted, but did not reach the point of generating tags.

Attempted to explore pretraining a model from scratch however, due to insufficient resources such as computational resources, I could not conduct a pretrained model for Bioinformatics data from a PubMed dataset that could improve the NER task and f1-scores over this dataset. But including an additional feature such as “chunks” for POS tags feature and additionally using domain specific tag information such as protein, gene, or even simpler binary tag classification such as B-BIO, B-CHEM, B-GENE, B-PROT, B-DNA may be helpful, similar to the case of B-PER, B-GEO etc.
I attempted Word Vector embeddings to further improve BioBERT, which seemed to be an even better approach. However this task requires domain-specific word embeddings as I came across this informative article: https://www.nature.com/articles/s41597-019-0055-0  

### Analysis:

The need for additional features such as the first letter is a capital letter, all letters are capital letters, the previous word is a hyphen, previous word or next word that is a number, previous word+ next word that is an alphanumeric string - all of which can act as better features for working with HMM.
The vocabulary consists of many rare words and clearly the model is not adaptable to unknown or rare words that are significantly different from words that do exist in vocabulary. it could be helpful to use a more domain specific trained model.
Domain specific knowledge mining tasks requires significant understanding of underlying structures whether the data is a biomedical textual data or as complex as a domain specific representation such as that of proteins, DNA bindings in a freeform text.
Could have explored a few more suggestions informed by the professor during class before the break, such as marking rare words with UNKNOWN or exploring BIOES, using PyTorch and word2vec.
Including an additional feature such as “chunks” for POS tags feature and additionally using domain specific tag information such as protein, gene, or even simpler tag classification such as B-BIO or B-CHEM or B-GENE, B-PROT, B-DNA may be helpful, similar to that of B-PER, B-GEO etc.

### References:
- <a href="https://bair.berkeley.edu/blog/2019/11/04/proteins/">Can We Learn the Language of Proteins?</a>
- <a href="https://web.stanford.edu/~jurafsky/slp3/8.pdf">Sequence Labeling for Parts of Speech and Named Entities</a>
- <a href="https://web.stanford.edu/~jurafsky/slp3/A.pdf">Hidden Markov Models</a>
- <a href="https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode">Viterbi algorithm</a>
- <a href="https://www.youtube.com/watch?v=DxLcMI-EMYI">Custom NER task using SPACY </a>
- <a href="https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Named_Entity_Recognition_BC5CDR.ipynb">Custom NER using BioBERT: </a> 
- <a href="https://www.nature.com/articles/s41597-019-0055-0">BioWordVec, improving biomedical word embeddings with subword information and MeSH</a>
- <a href="https://courses.engr.illinois.edu/cs447/fa2018/Slides/Lecture06.pdf">HMMs</a> 
- <a href="https://courses.engr.illinois.edu/cs447/fa2018/Slides/Lecture07.pdf">Sequence Labelling</a>
