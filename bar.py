from datetime import timedelta
from progress.bar import IncrementalBar

# Progress has issues on Windows
# https://github.com/verigak/progress/issues/58#issuecomment-471718558
def get_patched_progress():
	# Import a clean version of the entire package.
	import progress

	# Import the wraps decorator for copying over the name, docstring, and other metadata.
	from functools import wraps

	# Get the current platform.
	from sys import platform

	# Check if we're on Windows.
	if platform.startswith("win"):
		# Disable HIDE_CURSOR and SHOW_CURSOR characters.
		progress.HIDE_CURSOR = ''
		progress.SHOW_CURSOR = ''

	# Create a patched clearln function that wraps the original function.
	@wraps(progress.Infinite.clearln)
	def patchedclearln(self):
		# Get the current platform.
		from sys import platform
		# Some sort of check copied from the source.
		if self.file and self.is_tty():
			# Check if we're on Windows.
			if platform.startswith("win"):
				# Don't use the character.
				print('\r', end='', file=self.file)
			else:
				# Use the character.
				print('\r\x1b[K', end='', file=self.file)
	
	# Copy over the patched clearln function into the imported clearln function.
	progress.Infinite.clearln = patchedclearln
	
	# Return the modified version of the entire package.
	return progress

get_patched_progress()

class Bar(IncrementalBar):

    @property
    def total(self):
        return self.eta + self.elapsed

    @property
    def total_td(self):
        return timedelta(seconds=self.total)