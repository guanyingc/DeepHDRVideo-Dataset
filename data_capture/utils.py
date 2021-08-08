import os
import datetime
import inspect
import re

### IO Related ###
def make_file(f):
    if not os.path.exists(f):
        os.makedirs(f)
    #else:  raise Exception('Rendered image directory %s is already existed!!!' % directory)

def make_files(f_list):
    for f in f_list:
        make_file(f)

def empty_file(name):
    with open(name, 'w') as f:
        f.write(' ')

def read_list(list_path,ignore_head=False, sort=False):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists

def write_string(filename, string):
    with open(filename, 'w') as f:
        f.write('%s\n' % string)

def save_list(filename, out_list):
    f = open(filename, 'w')
    for l in out_list:
        f.write('%s\n' % l)
    f.close()

#### String Related #####
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def dict_to_string(dicts, start='\t', end='\n'):
    strs = '' 
    for k, v in sorted(dicts.items()):
        strs += '%s%s: %s%s' % (start, str(k), str(v), end) 
    return strs

def float_list_to_string(l):
    strs = ''
    for f in l:
        strs += ',%.2f' % (f)
    return strs

def get_datetime(minutes=False, seconds=False):
    t = datetime.datetime.now()
    dt = ('%02d-%02d' % (t.month, t.day))
    if minutes:
        dt += '-%02d.%02d' % (t.hour, t.minute)
    if seconds:
        dt += '.%02d' % (t.second)
    return dt
def raise_not_defined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)

