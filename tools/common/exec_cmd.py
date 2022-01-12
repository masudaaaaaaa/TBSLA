import subprocess
import sys
import fcntl
import os
import time


def execute_command(cmd):
  p = subprocess.Popen(cmd, shell=True)
  try:
    rv = p.wait()
    p.communicate()
    if rv != 0:
      print("Error executing :", cmd)
      sys.exit(rv)
  except subprocess.TimeoutExpired:
    p.kill()
    p.communicate()

def execute_command_pipe_stdout(cmd):
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
  try:
    rv = p.wait()
    out = p.communicate()
    if rv != 0:
      print("Error executing :", cmd)
      sys.exit(rv)
  except subprocess.TimeoutExpired:
    p.kill()
    out = p.communicate()
  return out


def nonBlockRead(output):
  fd = output.fileno()
  fl = fcntl.fcntl(fd, fcntl.F_GETFL)
  fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
  try:
    r = output.read()
    if r == None:
      return ''
    else:
      r = r.decode('utf-8')
      return r
  except:
    return ''

class TimeoutInterrupt(Exception):
  """Raised when process exceeds timeout."""
  pass

#https://stackoverflow.com/questions/3575554/python-subprocess-with-timeout-and-large-output-64k
def execute_command_timeout(cmdline, timeout=60):
  """
  Execute cmdline, limit execution time to 'timeout' seconds.
  Uses the subprocess module and subprocess.PIPE.

  Raises TimeoutInterrupt
  """

  p = subprocess.Popen(
   cmdline,
   bufsize = 0, # default value of 0 (unbuffered) is best
   shell   = True, # not really needed; it's disabled by default
   stdout  = subprocess.PIPE,
   stderr  = subprocess.PIPE
  )

  t_begin = time.time() # Monitor execution time
  seconds_passed = 0

  stdout = ''
  stderr = ''

  while p.poll() is None and seconds_passed < timeout: # Monitor process
    time.sleep(0.1) # Wait a little
    seconds_passed = time.time() - t_begin

    # p.std* blocks on read(), which messes up the timeout timer.
    # To fix this, we use a nonblocking read()
    # Note: Not sure if this is Windows compatible
    stdout += nonBlockRead(p.stdout)
    stderr += nonBlockRead(p.stderr)

  if seconds_passed >= timeout:
    try:
      p.stdout.close()  # If they are not closed the fds will hang around until
      p.stderr.close()  # os.fdlimit is exceeded and cause a nasty exception
      p.kill()          # Important to close the fds prior to killing the process!
                        # NOTE: Are there any other "non-freed" resources?
    except:
        pass

  returncode  = p.returncode

  return (returncode, stdout, stderr)
