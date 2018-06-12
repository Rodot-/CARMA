'''small library of custom context managers'''


class LoadingBar:


	def __init__(self, bar=False):

		self.is_running = True
		#self.mapping = '_.-^-.'
		self.mapping = '-/|\\'
		self.N = len(self.mapping)
		self.n = self.N*2
		self.n = 1
		self.blank = (' '*(self.n+2)).join('\r\r')
		self.done = False
		self.clock = 0
		self.prev_est = []
		self.percentage_est = 0
		if bar:
			self.set_as_bar()


	def print_loading_bar(self):

		N = self.N
		n = self.n
		while self.is_running:
			for j in xrange(N):
				sys.stdout.write(self.blank)
				for i in xrange(j,j+n):
					sys.stdout.write('('+self.mapping[i%N]+')')
				sys.stdout.flush()
				if not self.is_running:
					self.done = True
					break
				time.sleep(0.1)
		self.done=True

	def print_bar(self):

		self.N = 60
		self.blank = (" "*(self.N+20)).join("\r\r")
		while self.is_running:
			t_remain = self.est_remaining()
			for j in xrange(len(self.mapping)):
				sys.stdout.flush()
				string = "[" + "#"*self.n +self.mapping[j] + '.'*(self.N-1-self.n) + "] " + t_remain
				sys.stdout.write(self.blank+string)
				sys.stdout.flush()
				if not self.is_running:
					self.done = True
					break
				time.sleep(0.1)
		self.done = True

	def est_remaining(self):


		if self.percentage_est != 0:
			if self.n == self.N:
				return "Done"
			delta = time.time() - self.clock
			p = self.percentage_est
			est = delta/p*(1-p)
			self.prev_est.append(est)
			max_terms = max((min((self.N-self.n, self.n))+1)*4,31)
			min_terms = 30 # central limit theorem!
			if self.n > self.N / 2:
				min_terms = 1

			while len(self.prev_est) > max_terms:
				self.prev_est.pop(0)
			if len(self.prev_est) > min_terms:
				now = copy(self.prev_est)
				weights = np.linspace(0,2,len(now)) # weight the estimates linearly
				# for the weight function, I can chose anything as long as it integrates to 1
				# to make sure this happens, I can just divide by the sum
				weights[::] = np.exp(weights)
				weights /= (np.sum(weights)/len(now))
				est = int(np.mean(np.array(now)*weights))
				return "{}:{:02d} remaining".format(est/60, est%60)
		return "Estimating..."

	def set_as_bar(self):

		self.print_loading_bar = self.print_bar


	def update_bar(self, percentage):

		self.n = int(percentage * self.N)
		self.percentage_est = percentage
		self.est_remaining() # This will update our estimate faster


	def reset(self):

		self.clock = time.time()
		self.prev_est = []
		self.percentage_est = 0

	def __enter__(self):

		self.clock = time.time()
		self.prev_est = []
		self.percentage_est = 0
		self.done=False
		self.is_running=True
		print ""
		thread.start_new_thread(self.print_loading_bar, ())
		return self

	def __exit__(self, *args):

		self.n = self.N
		self.is_running = False
		while not self.done:
			time.sleep(0.001)
		#sys.stdout.write(self.blank)
		print ""
		sys.stdout.flush()
		self.n = 1
		self.clock = 0

