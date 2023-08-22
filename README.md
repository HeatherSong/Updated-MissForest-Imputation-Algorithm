# Updated-MissForest-Imputation-Algorithm

MissForest imputes missing values using Random Forests in an iterative fashion. By default, the imputer begins imputing missing values of the column (which is expected to be a variable) with the smallest number of missing values -- let's call this the candidate column. The first step involves filling any missing values of the remaining, non-candidate, columns with an initial guess, which is the column mean for columns representing numerical variables and the column mode for columns representing categorical variables. [1]

However, the current package that is available for installation is outdated as it does not comply with any sklearn version above 0.20.1. To solve this issue, I modified the original code [2] and now it is compatible with the newest sklearn version as of the day this file was created (1.2.2). 

**References**

[1] https://github.com/epsilon-machine/missingpy/blob/master/README.md
[2] https://github.com/epsilon-machine/missingpy/tree/master/missingpy
