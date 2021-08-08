import datetime, time
import os
import utils

class Logger(object):
    def __init__(self, save_dir):
        self.start_time = time.time()

        datetime = utils.get_datetime(minutes=True)
        logdir = os.path.join(save_dir, 'logdir')
        utils.make_file(logdir)
        self.logfile = open(os.path.join(logdir, 'log_%s.txt' % datetime), 'w')

    def reset_timer(self):
        self.start_time = time.time()

    def print_write(self, strs):
        h, m, s = self.get_runtime()
        strs = '[%02dh:%02dm:%02ds] %s' % (h, m, s, strs)
        print(strs)
        self.logfile.write(strs + '\n')
        self.logfile.flush()

    def print_params(self, params):
        if type(params) is not dict:
            params = vars(params)
        params_str = utils.dict_to_string(params)
        self.print_write(params_str)

    def get_runtime(self):
        seconds = time.time() - self.start_time
        hours, seconds = seconds // 3600, seconds % 3600
        minutes, seconds = seconds // 60, seconds % 60
        #time_elapsed = (time.time() - self.start_time) / 3600.0
        return hours, minutes, seconds

    def get_elapsed_seconds(self):
        seconds = time.time() - self.start_time
        return seconds

    def close(self):
        self.logfile.close()
