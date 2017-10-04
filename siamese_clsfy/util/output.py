import time


def title(title):
    def title_decorate(func):
        def func_wrapper(*args, **kwargs):
            print '##################################################\n' \
                  '#                                                #\n' \
                  '#{:^48}#\n' \
                  '#                                                #\n' \
                  '##################################################\n'.format(title)
            func(*args, **kwargs)
            print '\n\n'
        return func_wrapper
    return title_decorate


def subtitle(title):
    def subtitle_decorate(func):
        def func_wrapper(*args, **kwargs):
            print '          ##############################\n' \
                  '          #{:^28}#\n' \
                  '          ##############################\n'.format(title)
            func(*args, **kwargs)
            print '\n\n'
        return func_wrapper
    return subtitle_decorate


def timer(name):
    def time_decorate(func):
        def func_wrapper(*args, **kwargs):
            start = time.time()
            re = func(*args, **kwargs)
            timestamp = time.time() - start
            print '++++++++++++++++++++++++++++++++++++++++++++++++++'
            print name, ': {:.3f} sec'.format(timestamp)
            return re
        return func_wrapper
    return time_decorate


def avg_timer(name):
    def time_decorate(func):
        def func_wrapper(*args, **kwargs):
            start = time.time()
            for _ in range(0, 20):
                re = func(*args, **kwargs)
            timestamp = (time.time() - start) * 1.0 / 20
            print '++++++++++++++++++++++++++++++++++++++++++++++++++'
            print name, ': {:.3f} sec'.format(timestamp)
        return func_wrapper
    return time_decorate