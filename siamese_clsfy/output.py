import time


def title(title):
    def title_decorate(func):
        def func_wrapper():
            print '##################################################\n' \
                  '#                                                #\n' \
                  '#{:^48}#\n' \
                  '#                                                #\n' \
                  '##################################################\n'.format(title)
            func()
            print '\n\n'
        return func_wrapper
    return title_decorate


def timer(name):
    def time_decorate(func):
        def func_wrapper(param):
            start = time.time()
            re = func(param)
            timestamp = time.time() - start
            print '++++++++++++++++++++++++++++++++++++++++++++++++++'
            print name, ': {:.3f} sec'.format(timestamp)
            return re
        return func_wrapper
    return time_decorate


