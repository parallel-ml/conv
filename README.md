# parallel-ml

## cnn
Contains module for convolutional layer split. This directory implements with Python
module. Run test or file under root directory. 

### run tests
Run all test files
```angular2html
nosetests
```
Run specific test module
```angular2html
nosetests cnn/tests/test_models.py:test_vgg16
```

## fc
Contains module for fully connected layer parallelization. This directory is not
modulized. Run each file in its directory.