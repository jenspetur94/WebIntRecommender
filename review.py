class Review():
	def __init__(self, userId, itemId, rating):
		self.userId = int(userId)
		self.itemId = int(itemId)
		self.rating = int(rating)
