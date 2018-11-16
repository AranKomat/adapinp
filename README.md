# Adaptive Input
Unofficial implementation of Adaptive Input in PyTorch

Combined with Adaptive Softmax, Adaptive Input drastically decreases the number of parameters of your model.

It is also known that Adaptive I/O results in lower ppl in rare words, since the traditional I/O layers underfit to infrequent tokens.

I/O tying is not supported yet, though it's quite simple to do that. 

For more details, please refer to the paper: Adaptive Input Representations for Neural Language Modeling by Alexei Baevski, Michael Auli https://arxiv.org/abs/1809.10853. 

Your can import Adaptive Softmax as AdaptiveLogSoftmaxWithLoss from torch.nn.modules.adaptive. 

For using adaptive I/O, you need to preprocess your text (or more generally sequence) dataset, so that the number assigned to each token is in the order such that the more frequently occuring token is assigned lower number.
