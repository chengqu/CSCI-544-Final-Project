% Modular Deep Encoder-Decoders 
% Group 40: Arnold, Gurunat, Huang, & Qu 
% \today

\begin{abstract}
In this short paper we propose an approach to transfer learning using rich
vector embeddings. The suggested technique can be applied to any supervised
task, and it handles multiple sources and changing sources of data without the
need for retraining. To verify our ideas, we apply our ideas to the task of
text-classification.
\end{abstract}

# Introduction
Our goal is to generate rich vector embeddings from articles to classify
them into predefined categories. Then, we want to extend our
model with an additional source of data: the title of the videos. To handle this
new knowledge source without retraining our previous model, we suggest to
generate a new embedding that will be used to modify the original one. This
combined embedding will then be used for the classification task.

More formally, we want to jointly learn a set of *encoder* functions $\{E_i\}_0^N$
mapping samples $x \sim \chi_i$ from a set of data distributions $\{\chi_i\}_0^N$ to a
fixed-sized vector embedding $V$.

$$ \forall i \in [0; N]: E_i(x): \chi_i \rightarrow V_i \in \Re^M$$

$$V = \sum_i^N V_i$$

The embedding $V$ is then fed into a decoder function $D$ which in the case of
classification learns a mapping from the vector space of $V$ to the label space
$L$.

$$ D(v): \Re^M \rightarrow L \in \Re^D$$

Finally, we want to extend the set of encoders by learning a new encoder $E_{N+1}$ which
handles samples from a different dataset $\chi_{N+1}$, without retraining our
trained decoders and encoders.

$$ E_{N+1}(x): \chi_{N+1} \rightarrow V_{N+1} \in \Re^M$$

Although we could have used any kind of mapping, we chose to use deep learning
algorithms, as they easily learn hierarchical representations and have been
known to highly outperform other statistical techniques on natural language
tasks. Learning embeddings has been previously done by [[1]](References),
although our proposed contribution is slightly different. [[1]](References) used
deep autoencoders to obtain a better initialization of the parameters of their
model. Our approach instead is much closer in spirit to the work of Sutskever
[[3]](References) and Vinyals [[4]](References). They both use encoders on a
sequence of data to generate a *thought-vector* which will then be decoded in
the desired terget sequence. In some sense, our proposal adds the transfer
learning component to their contribution. This approach is also similar to the
work of Karpathy & al [[5]](References) where they mapped images to their captions
with embeddings.


# Method

## Materials

We used the freely available dataset from
[https://github.com/ParallelMazen/SaudiNewsNet](https://github.com/ParallelMazen/SaudiNewsNet)
The dataset contains a total of 31,030 Arabic newspaper articles, with title, author, 
date, url and content in each entry. Article objects are reprensented in json format, 
with UTF-8 encoding. We wrote a script to download the data from github repo, unzip 
each file, and read them in as json objects. The function can also give out content 
by key values such as title, author and etc. 


## Procedure
The procedure is quite straight-forward. While part of the team worked on
building our tailored dataset, the other half worked on the model definition
and training procedure. 

Once every pre-requisite is available, we trained the first encoder $E_1$
(implemented as a recurrent neural network) to build the embedding $V_1$. Since
we only dealt with a relatively simple classification task, our
decoder $D_1$ was simply a fully connected multi-layer perceptron. They were 
jointly trained end-to-end by propagating the gradients through the embedding
from $D_1$ to $E_1$.

The second training step was to train the second encoder $E_2$. Again, we
also performed training end-to-end, but specifically did not propagate the
gradients through $E_1$. 

## Evaluation



# Results


# Discussion




# References
Part of the relevant literature review. More literature was involved for the deep learning part.

1. Supervised Representation Learning: Transfer Learning with Deep Autoencoders, Zhuang & al, http://ijcai.org/Proceedings/15/Papers/578.pdf

2. Transfer Learning via Dimensionality Reduction, Pan & al,

    https://www.cse.ust.hk/~jamesk/papers/aaai08.pdf

3. Sequence to Sequence Learning with Neural Networks, Sutskever & al,

    http://arxiv.org/abs/1409.3215

4. Grammar as a Foreign Language, Vinyals & al, http://arxiv.org/abs/1412.7449

5. Deep Fragment Embeddings for Bidirectional Image Sentence Mapping, Karpathy &
   al, https://cs.stanford.edu/people/karpathy/nips2014.pdf
