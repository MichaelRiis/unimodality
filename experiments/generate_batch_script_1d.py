import numpy as np

output_filebase = "run_1d_batch"

base_str = "python run_1d.py"
# function_classes = ['gaussian', 'student_t', 'tukey']
function_classes = ['tukey']
max_functions = 100

lines_per_job = 10
time_per_line = 25

# generate run line for each run
lines = []
for function_class in function_classes:
	for idx_fun in range(max_functions):
		lines.append(base_str + " --function_class " + function_class + " --function_idx %d" % idx_fun )


# computer number of jobs and time per job
num_lines = len(lines)
num_jobs = np.ceil(num_lines/lines_per_job).astype(int)
hours_per_job = np.ceil(time_per_line*lines_per_job/60).astype(int)
assert(hours_per_job < 24)


print('%d jobs needed with a maximum run time of %4.3f hour(s)' % (num_jobs, hours_per_job))

# split in job files

for idx_job in range(num_jobs):

	output_file = output_filebase +  "%d"%idx_job + '.sh'

	print(100*'-')
	print('Job %d: filename = %s ' % (idx_job, output_file))
	print(100*'-')

	f  = open(output_file, 'w') 



	f.write('#!/bin/bash\n')
	f.write('#SBATCH --time=0-%02d:00:00\n' % hours_per_job)
	f.write('#SBATCH --mem-per-cpu=1000\n')
	f.write('\n')

	# move first part of 'lines' to 'current_lines'
	current_lines = [lines.pop(0) for idx_line in range(lines_per_job)]

	for line in current_lines:
		f.write(line + '\n')

	f.close()

	print('Done.')
	print('\n\n')





