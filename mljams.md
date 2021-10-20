# ML-JAMS

Matt, Julian, Athir, Mariam, and Sam learn about machine learning.

## Syllabus

The syllabus mostly follows the (in)famous
[Machine Learning class](https://www.coursera.org/learn/machine-learning),
but at a slower pace.

Week | Instructions        | Topic
-----|---------------------|-------------------------------------------------
 01  | [week 01](#week-01) | What is machine learning?
 02  | [week 02](#week-02) | Linear algebra review
 03  | [week 03](#week-03) | Linear regression with multiple features I: Gradient descent
 04  | [week 04](#week-04) | Linear regression with multiple features II: Direct solution
 05  | [week 05](#week-05) | The classification problem
 06  | [week 06](#week-06) | Logistic regression
 07  | coming soon...      | Neural networks: Overview
 08  | coming soon...      | Neural networks: Details
 09  | coming soon...      | Neural networks: Implementation
 10  | coming soon...      | Neural networks: This was only the beginning
 11  | coming soon...      | Machine learning methodology I:
 12  | coming soon...      | Machine learning methodology II:
 13, 14 | coming soon...   | Support vector machines?
 15  | coming soon...      | Unsupervised learning: Clustering
 16  | coming soon...      | Unsupervised learning: Dimensionality reduction
 17  | coming soon...      | Unsupervised learning: Anomaly detection
 18  | coming soon...      | Unsupervised learning: Recommender systems
 19--22 | coming soon...   | More practical advice, applications, etc.

Additional topics (available on request, also suggestions welcome):

* Reinforcement learning
    * Reinforcement learning for finite worlds
    * Reinforcement learning for infinite worlds
    * Deep reinforcement learning
    * Reinforcement learning and neuroscience
    * Reinforcement learning and video games
    * Reinforcement learning and robotics
* ML (/AI) and society
    * World-record machine learning systems
    * The power and limits of scaling machine learning
    * The ethics of modern AI systems in the world (user profiling,
      lethal autonomous weapons, etc.)
    * Technical approaches to AI ethics (interpretability, etc.)
    * Generally intelligent and superintelligent ML and AI systems
    * Technical approaches to general AI safety


## Week 01

**What is machine learning?**

*Abstract:*

This week we dive into the ML class and watch half of the week 1 lectures.
Optional: Install ML software, answer some discussion questions, and dive
deeper into the context of machine learning.


*Basic [2h]:*

1. [5m] Sign up for the ML class on coursera
   (https://www.coursera.org/learn/machine-learning).
   Skip past the welcome info and navigate to the week 1 materials.

2. [30m @1.5x] Watch the four videos in the "Week 1: Introduction"
   section at 1.5x or faster.
   Learn not to bother with the "readings" (they just recapitulate the
   videos or contain useless coursera administrivia).

3. [45m @1x] Slow down to 1x speed (or faster) and watch the videos in
   the "Week 1: Model and Cost Function" section.
   Pay attention to the new terminology.
   Try to follow the visualisations closely. If it's not clear what the
   picture is showing you, stop and think it through, or ask for help.

4. [40m @1x] Watch the videos in "Week 1: Parameter learning".
   Pay particular attention to the intuition for gradient descent video
   (video 2).
   Pay attention to the details of the linear regression example (video 3).
   In video 3, even though we won't need to do this kind of math for the
   rest of the course, I think this example is within reach of high-school
   math and it would be helpful to make the connection. So: Try to work
   through this derivation yourself. If you're not ready for this, let me
   know, and let's work through it together.


*Coding [30m]:*

1. [10m] Install Python's scientific stack (numpy, scipy, matplotlib), or
   Octave, or Julia, or R (whichever is your preference; language learning
   time not included in this course).

2. [10m] Install Jupyter notebook and get a notebook up and running with a
   kernel in the language of your choice (you'll have to find instructions).

3. [10m] Make a notebook that just plots a straight line given a slope term
   and a y-intercept term. This is just a "hello world" plotting task: No
   machine learning required.


*Discussion [30m]:*

Post your responses to one or more of the following questions in the chat:

1. The linear regression problem is probably familiar to you from high-school.
   Did you also learn any other methods of finding a line of best fit (aside
   from gradient descent)?
   We'll revisit this in the coming weeks.

2. The "welcome" lecture says "There was a realization that the only way to
   do these things was to have a machine learn to do it by itself". To what
   extent do you think the "learning" analogy captures what is going on in
   applying the gradient descent algorithm to find a line of best fit in the
   linear regression example?

3. Does the linear regression example fit into "supervised" or "unsupervised"
   learning? What is the difference between "supervised" and "unsupervised"
   learning?

4. We are quite familiar with the process of "human learning". Does this
   process fit into the framework of machine learning? And is human learning
   typically "supervised", "unsupervised", or perhaps both, or perhaps neither?
   Consider both learning as a child and learning in school.

*Extra [1h15m]:*

1. [30m] Read the introduction, overview, and "history and relationship to
   other fields" section of the Wikipedia page on machine learning
   (https://en.wikipedia.org/wiki/Machine_learning).

2. [30m] Read section 7 "Learning Machines" (pdf pages 23 to 29) of Alan
   Turing's 1950 paper "Computing Machinery and Intelligence"
   (https://www.cs.mcgill.ca/~dprecup/courses/AI/Materials/turing1950.pdf).
   This is the paper in which Alan Turing proposed the famous "imitation
   game" (in the first few sections). The whole paper is worth reading in
   your own time.

3. [15m] Watch the video "Building the Software 2.0 Stack" by Andrej
   Karpathy (director of artificial intelligence at Tesla)
   (https://www.youtube.com/watch?v=y57wwucbXR8).

## Week 02

**Review of linear algebra**

*Abstract:*

This week we will explore the wonderful world of the mathematics of arrays of
numbers. Vectors and matrices are the central objects of machine learning.
This week, we're aiming to get to know them as well as we know our old friend
the number line.

Unfortunately Andrew Ng's introduction to matrix algebra is pedagogically
bankrupt. So, we're going to skip most of those videos and go instead for a
world-class matrix algebra education from the living legends, Gilbert Strang
from MIT opencourseware and 3blue1brown from youtube!

*Basic [2h]:*

1. [15m @1.5x] Watch the first two videos from the "Week 1: Linear Algebra
   Review" section, at 1.5x speed or faster, but **also solve the quiz
   questions**.
   This much is necessary to introduce the terminology. Ask if anything is
   unclear, but I guess it should be straight forward so far.

2. [1h @1.0x] Watch the first five chapters of 3blue1brown's "Essence of
   Linear Algebra" course. Pause to consider each of the examples---follow the
   math that goes with the examples, and make sure it all makes sense, in
   detail! Write it down if you have to! There is plenty of time.

   * [Chapter 1: Vectors](https://www.youtube.com/watch?v=fNk_zzaMoSs)
   * [Chapter 2: Linear Combinations, Span, and Basis Vectors](https://www.youtube.com/watch?v=k7RM-ot2NWY)
   * [Chapter 3: Linear Transformations and Matrices](https://www.youtube.com/watch?v=kYB8IZa5AuE)
   * [Chapter 4: Matrix Multiplication as Composition](https://www.youtube.com/watch?v=XkY2DOUCWMU)
   * [Chapter 5: Three-dimensional Linear Transformations](https://www.youtube.com/watch?v=rHLEWRxRGiM)

3. [45m @1.0x] Watch this lecture from Gilbert Strang's "Introduction to
   Linear Algebra" class:
   [Multiplication and Inverse Matrices](https://www.youtube.com/watch?v=FX4C-JpTFgY).
   Pay close attention to the first 22 minutes (about matrix multiplication)
   and skim the rest (on inverses, don't worry if it doesn't all make sense
   just yet).


*Discussion [10m]:*

Post your responses to the following question in the chat:

1. We'll spend very little time in 2D and 3D during this course. To what extent
   do you think the visual examples of low dimensional space are useful for
   understanding real machine learning systems, where vectors may have hundreds,
   hundreds of thousands, or these days even hundreds of billions of dimensions?


*Exercises [1h30m]:*

Exercises are not meant to be easy or enjoyable. Apologies in advance.

1. [10m] Invent two four-by-four matrices using only the numbers 0, -1, 1, 2,
   3, 4, 6, and 12. No column or row should have more than two of the same
   number. Try to make one row/column linearly dependent in each matrix.

2. [15m] Multiply the two matrixes, the old-school way: Compute the 16
   elements of the new matrix, one at a time, as the dot products of the
   corresponding rows and columns.

3. [10m] Multiply the two matrices again, this time using the second method
   from Gilbert Strang's lecture: Do four vector-matrix multiplications with
   the columns of the second matrix.
   Of course, you should get the same answer.

4. [10m] Multiply the two matrices again, this time using the third method
   from Gilbert Strang's lecture: Do four vector-matrix multiplications with
   the rows of the first matrix.
   Of course, you should get the same answer.

5. [10m] Again. This time compute the four column--row products like in the
   fourth example from Gilbert Strang's lecture.
   Of course, you should get the same answer.

6. [10m] One more time: Block multiplication. Compute the matrix product by
   dividing the matrices each into four 2x2 blocks.

7. [10m] Did I say last one? This one doesn't count, the computer does the
   hard work! Load your programming environment. Learn how to enter the
   matrices and print/visualise them. Learn how to multiply matrices, and
   compute the answer. Check that the computer didn't make a mistake.

8. [15m] OK, last one, really this time. For an extra challenge: Think of
   the matrices as 4-dimensional linear transformations of the
   four basis vectors (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), and
   (0, 0, 0, 1), extending the ideas from the 3blue1brown videos. What
   transformation do your matrices involve? Can you compute the composition?





## Week 03

**Linear regression with multiple features I: Gradient Descent**

*Abstract:*

This week we will extend our simple linear regression model to permit multiple
input features, and we will extend our learning algorithms to put weights on
all of the features.

The videos are quite light this week, taking up to one hour. There are three
choices for what to do with the spare time (choose one or more of these, and
note there will be another hour for the same three activities next week):

* A. Take a numerical linear algebra tutorial in your language of choice (the
   coursera course includes an octave/matlab tutorial, there should be plenty
   of numpy/julia/etc. tutorials online).
* B. Complete some programing exercises (see below).
* C. Complete some discussion questions and math exercises (see below).


*Basic [1h]:*
   
The math is starting to get a little more involved. I've allocated extra
time so that you have time to pause or slow down to follow the examples if
it's too fast. Don't gloss over the equations: Think them through!

1. [15m @1.0x] Watch the first video from "Week 2: Multivariate Linear
   Regression", at 1.0x speed or faster, and solve the quiz question.
   This video sets the notation and it's particularly important to follow.
   You might like to take some notes. Ask if anything is unclear.

2. [15m @1.0x] Watch the second video and solve the quiz question. This second
   video discusses the update rules for gradient descent in the new setting.
   Don't worry about the detailed derivation of these rules yet (see optional
   exercises) but do follow closely enough that you understand where the
   different parts of the final update rule come from. **The most important
   thing is to be able to relate it back to the intuition about gradient
   descent from week 1.** Once you have glimpsed that, you are golden. Ask if
   unclear.

3. [30m @1.0x] Watch the remaining three videos and solve the exercises.
   The details of these videos are not as important, but the high-level ideas
   are very important.

*Coding Exercises [1h+]:*

Coding exercises can take an open-ended amount of time. I've tried to offer
clear steps, but if you get stuck or get carried away... well, you know how
it is.

1. [5m] Start a new jupyter notebook in your chosen programming language.

2. [10m] Generate a random dataset of (1) 100 random 2d vectors with components
   in the range [0, 1], and (2) 100 labels given by the average of the two
   components of each of the random vectors
   (probably something like `y = 0.5 * x[:, 0] + 0.5 * x[:, 1]`).

3. [15m] Figure out how to make a 3d plot of the dataset. Don't worry about the
   details like axis labels, just get the visualisation.

4. [30m] This dataset can be fit by a multivariate linear regression model,
   without a bias term.
   * Implement the gradient descent algorithm starting at (0.1, 0.8), using the
     update equations from the lecture (see reading page), some small learning
     rate, and some large number of iterations.
   * Does the algorithm eventually find the (0.5, 0.5) optimal solution?
   * Plot (in 2d) the trajectory of the parameter vector throughout the
     learning algorithm. For bonus marks, add a contour plot of the cost
     function in the background.


*Discussion [30m]:*

1. Where do features come from? What difference does the choice of features
   make to the success of the regression model? Can you think of some examples
   to illustrate your answer?

2. In the final video we discussed adding more and more complex features to
   the model to help us capture the relationship between the input and the
   output variables. Are we still doing *linear* regression when some of these
   features are nonlinear functions of the original features?

3. Follow-up question: Is there any disadvantage to having *too many*
   features, or can we just continue to add every feature we can think of?
   If the features don't help, won't the gradient descent algorithm just set
   their weight parameters to zero and subsequently ignore them?

4. In light of your discussion, what are some good principles for selecting
   features for a linear regression model?


*Math exercises [30m]:*

1. Where did those update rules come from?
   Return to video 2 or the reading afterwards and look up the cost function.
   Flex your calculus muscles by deriving the partial derivative with respect
   to theta 0 and then also the partial derivative with respect to theta 1
   (the rest are like the latter, so this is enough). Ask if you need a hand.


*Extra*

* Later in the course, we'll learn about *neural networks*. At the basic
  level, these are large circuits of simple linear regression models connected
  in parallel (layers) and in series (depth), to use an electrical engineering
  analogy. The intuitions we have developed for linear regression so far will
  be useful when we come to study deep learning (training neural networks).
  Hold on to it!


## Week 04

**Linear regression with multiple features II: Direct solution**

*Abstract:*

This week we will conclude our study of the basic linear regression model by
deriving the 'normal equations', which is a direct mathematical solution to
the problem we have so-far used gradient descent to solve.

Once again the videos are pretty light this week. This time they take around
30 minutes. With the spare time, try the mathematical challenge below, and
then return to complete the week 3 activities which you haven't already
completed.

*Basic [30m]:*

The math this week is pretty serious, in particular we'll be doing some
calculus with matrices and vectors directly.

They don't teach you this in school but it's super useful for deriving
gradient descent update rules. Fortunately, nobody actually has to do this in
practice, because there are software tools that will differentiate your models
for you when you actually build them with code.

So this week, you can follow the mathematics lightly. Well, at least *try* to
follow the calculations at an intuitive level, and ask if things are unclear.


1. [30m @ 1.0x] Watch the two videos from "Week 2: Computing Parameters
   Analytically", at 1.0x speed or faster, and solve the quiz questions.

*Math challenge [30m]:*

While it's beyond our scope to derive the normal equations for *multivariate*
linear regression, we can *totally* handle the derivation for *univariate*
linear regression.

1. [10m] Return to the readings from week 1 and write down the single variable
   linear regression cost function in terms of the two parameters. It might
   also help to note the definition of each of the other terms, including the
   data and labels, the number of data points, etc.
2. [10m] If you haven't already, differentiate the cost function with respect
   to each of the two parameters. If you've completed the optional math
   activity from last week, this will be similar. If you're stuck, reach out.
3. [10m] Now you should have two equations: one for each derivative. Set the
   derivatives to zero, and attempt to solve this system of linear equations
   with the techniques you learned in high-school, or by writing the system as
   a matrix and using some linear algebra.

In the end, you should be able to find an equation for the parameters in terms
of the data and labels. This is the normal equation for single-variable linear
regression. Congratulations!


*Remaining time:*

Return to the uncompleted week 3 activities.


*Advanced:*

This concludes our study of linear regression. However, this is not the end
of the topic. Linear regression is a perfect place to study the topics of
(1) *regularisation*: where the cost function is altered to prevent
    overfitting), and
(2) *statistical machine learning*: where the techniques of machine learning
    are interpreted in terms of statistical inference.

A great place to learn more about these topics using the linear regression
model is **the first three weeks** of the [Columbia University EdX course on
Machine Learning](https://www.edx.org/course/machine-learning).


## Week 05

**The classification problem**

*Abstract:*

This week we'll shift temporarily from the *regression problem* to the
*classification* problem; another classic problem studied in supervised
machine learning.
Classification refers to learning functions whose output is not a number
(like in regression) but a category (like a word, a colour, a label).

The corresponding week of the Coursera course doesn't spend enough time on
the principles, so we'll spend this week getting a high-level overview of
various classification techniques, and next week diving into the details of
logistic regression from Coursera.

*Basic:*

1. So far, we have considered only *the regression problem*. This week, we
   will explore *the classification problem*. Watch this
   [youtube video](https://www.youtube.com/watch?v=G_0W912qmGc).
2. Answer the following quiz questions:
   * The video mentioned the 'output' type of the functions we are learning.
     Classification and regression functions share a common input type. What
     is the input type?
   * For each of the following supervised learning problems, would it fit
     into classification or regression?
     * Predicting apartment prices given floor area and location data.
     * Predicting whether an image depicts
       [raw chicken or Donald Trump](https://www.kraftfuttermischwerk.de/blogg/wp-content/uploads2/2016/03/HUYPLXW.jpg)
     * Recognising a handwritten digit in an image.
     * Counting the number of human faces in an image.
   * For each of the above, what is the input type, and what is the output
     type?
3. Now we aim to get very high-level overview of several approaches to the
   classification problem. The emphasis this week is on exploring a broad set
   of approaches (the specific details are not so important this week).

   Watch the following videos:
   * [K-nearest neighbours](https://www.youtube.com/watch?v=HVXime0nQeI):
     A geometric approach to classification.
     Classify new points based on the most similar examples from the dataset.
     (Watch until about half way, when he starts talking about heatmaps.)
   * [Decision trees](https://www.youtube.com/watch?v=_L39rN6gz7Y):
     Automatically learning a simple rule-based flow-chart to classify new
     examples.
   * [Random forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ):
     Aggregating a population of decision trees together to get better
     classifications through voting.
   * [Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA):
     A classifier based on a very simple, 'naive' probabilistic model of
     the data. Famously effective for email spam filtering.
   * Perceptron and perceptron algorithm (TODO).
   * [SVMs](https://www.youtube.com/watch?v=efR1C6CvhmE):
     A linear classification approach based on finding a robust
     decision-boundary.
   * [Logistic regression](https://www.youtube.com/watch?v=yIYKR4sgzI8): 
     A popular approach that uses linear-regression-like techniques to solve
     the classification problem. We will dive into the details next week.
4. Most of these examples have been designed for 'binary classification',
   that is, for deciding between two outputs. But often we have more outputs.
   There are several trchniques that can convert binary classifiers into
   multi-class classifiers.
   Return to coursera to watch the video in "Week 3: Multiclass
   classification".


*Coding exercises:*

1. Most of the approaches to classification we have met will be available
   in your chosen machine learning toolkit. Find a tutorial or the
   documentation and follow along. Pick one dataset to test all the
   approaches on. Which gives the highest accuracy?

   (A legendary and simple dataset for classification is the "iris database",
   containing measurements from three types of iris flowers. The database is
   available in some statistics library and all over the internet. If your
   tutorial doesn't suggest a dataset, you could try this one.)


*Discussion:*

1. Logistic regression is a classification algorithm, even though the name has
   the word regression. It's not a mistake: In a way, we are using regression
   to perform classification. What is the regression problem inside this
   classification problem?

2. The output of a classifier function is sometimes represented as a number.
   It could be 0 or 1 for binary classification (e.g. distinguishing cats and
   dogs)
   or it could be a number from 0 to k for multiclass classification (e.g.
   recognising digits from 0 through to 9).
   Since these outputs are always subset of the real numbers, it seems like it
   should be possible to use *any* regression algorithm, even perhaps linear
   regression, to try to output the class label directly.
   However, this is usually considered a pretty bad idea. What's the problem?



## Week 06

**Logistic regression (for classification)**

*Abstract:*

This week we'll dive into the details of a particular classification approach:
*Logistic regression*.
This gives us a chance to see another example (besides linear regression) of
the model, cost function, gradient descent algorithm pattern that underlies
much of machine learning.
We will also attempt to implement a simple classification algorithm from
scratch (in contrast to last week where we just called into libraries).

*Basic:*

This week, we'll stick mainly to the coursera videos. Apologies in advance.

1. Watch the three videos from the "Week 3: Classification and Representation"
   section. These videos formulate the approach of using 
   
   TODO: WHAT ARE YOU LOOKING FOR?
2. Watch the three videos from the "Week 3: Logistic Regression Model" section
   and the three from "Week 3: Logistic Regression Model".

   TODO: WHAT ARE YOU LOOKING FOR?
3. TODO: WHAT ELSE?

*Coding exercises:*

Last week, we relied on machine learning toolkits to take care of the details
of the logistic regression learning algorithm. This week, we've seen most of
those important details in the lecture and so we should be able to put
together a simple implementation of this model.

1. TODO: Step-by-step instructions for the dataset,


*Advanced:*

* There is more on classification later: Week 7 on coursera is on the details
  of SVMs. It's also possible to use neural networks for 'probability
  regression'.
* The Columbia University EdX course has some nice material on classification
  algorithms, including probabilistic foundations ('Bayes optimal classifier')
  See week 4 and part of week 5.


## Week 07, 08, 09, 10

**Neural Networks**

There are two time-light but detail-heavy weeks on neural networks in the
coursera course. I prefer to skim details for a first course, and focus on
high-level understanding. But we can look into some details too if you like!
There is plenty to study here, and I think if we want to go deeper,
particularly into implementing neural networks, we should do a whole
follow-up course. Anyway, let's make these four weeks count!

1. Maybe we should start with some of Andrew Ng's intro, then watch the full
   3b1b series (it's pretty short).
   At some point I'll be able to share my recent talk recording, which could
   be extra overview-level information.
2. We could spend one week going through the coursera details on forward and
   backward passes.
3. We could spend one week implementing a simple neural network from scratch
   on toy data (math details involved, good way to master at low-level,
   quite challenging)
   OR
   we could spend the same time implementing a larger network in a standard
   framework on some more substantial data (higher level, more hand-wavy, but
   avoids low-level details).
4. I would like to add a week with a tour of some more advanced models (CNNs
   for speech/image processing, RNNs and transformers for language, Graph
   neural networks, GANs for deepfakes, etc.

More: A whole deep learning course.

## Week 11, 12

**Machine learning methodology**

* Remember to link to "optimizer's curse" (AIMA pp. 630, Smith and Winkler
  2006) when we talk about overfitting.
* But also talk about double descent...! Deep learning is special?
* Remember to have a discussion about where cost functions come from!




## Notes

* Richard Ngo lecture on machine learning, with a view towards AI safety
  https://youtu.be/lFRez9TFY5k
  (I don't agree with all of the framing and I think it's not suuuper
  accessible, but it was not thaaat bad)


* Find my lecture on artificial and biological neural networks, and see if the
  first part would be a useful introduction to the former.

* There should be a greater emphasis on the general, formal 'regression'
  problem, and even the 'function estimation' problem, in the first week.
  Even then we can talk about things like NFL, simplicy, etc.
  I guess I was lead astray. Curse you Andrew Ng!

* https://sebastianraschka.com/blog/2021/dl-course.html
