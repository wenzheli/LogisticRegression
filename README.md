Java Implementation for LogisticRegression 
==================

Logistic Regression for Java. 

Include those features. 
- Binary classification (we assume response variable is 0,1) 
- Stochastic Gradient Ascent.  (online learning for optimizing maximum likelihood)
- Sparse Feature Vector  (For high dimensional sparse data set, we can represent the feature vector as hashmap, to speed up the learning. 

Example code for using logistic regression for stochastic optimization. 

<pre>
<code>
   // generate training data
   double[][] x = new double[][]{
      {1,3}, {4,6}, {2,5}, {6,10}, {1,0.5}, {5,1}, {5,2}, {6,3}
   };

   int[] y = new int[]{1,1,1,1,0,0,0,0};
   
   LogisticRegression lr = new LogisticRegression(2);  // since our training data hss 2 dimension (excluing response variable)
   
   // set the number of paths. Each path will iterate all the training data
   for (int itr = 0; itr < k; itr++){
      for (int i = 0; i < x.length; i++)
        // update the weights. 
        lr.updateWeights(x[i], y[i]);
   }
   
   // after finish training, you can test your classifiers. 
   
   // generate test data
   double[][] test = new double[][]{ 
        {1,5},{6,4},{5,0}
   };
   
   for (int i = 0; i < test.length; i++){
      // get the probability. For (0,1) response variable, we assume this probability is for case "1"
      double prob = lr.eval(test[i]);  
      // continue doing sth.....
   }
   
  
</code>
</pre>
For derivation, please take a look at the LogisticRegression.pdf
