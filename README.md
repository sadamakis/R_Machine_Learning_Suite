# R Machine Learning Tools

## Introduction
The R Machine Learning Tools contains a set of R scripts geared towards the development of machine learning models.
The toolkit contains 6 parts.
1. Preprocess and transformation
2. General Feature selection function
3. R machine learning functions
4. H2O machine learning functions
5. Evaluation
6. Post Process

## Initial Engagement
- The ... team will manage the repository.
- The repository will be created in the ... project in Stash.

#### Reference
- region - the region running the model. Examples: us, uk
- type - the type of portfolio being analyzed. Examples: app3, app1, cnp, cash
- library - the library implementing the model. Examples: h2o
- algorithm - the model algorithm. Examples: RandomForest, NeuralNetwork, XGBoost
- number - the sequence number for the region/type/library combination
- modelFile - the name of the trained model artifact
- contractFile - the name of the contract file

## Git instructions
### Configuration
git config --global user.name "Your Name"
git config --global user.email "you@yourdomain.example.com"
git config --list

### Start
git clone ... (add git path)
git init
git pull

### Update contents and push to remote repo
git add ./README.md
git status

git reset ./README.md
git status

git commit -m "Modified document"

### Publish your change
- `git add <filename>` - Add any changed files to git
- `git commit -m"<some comment here>"` - Commit the changes to the repository
- `git push` - Push the commits to Stash
