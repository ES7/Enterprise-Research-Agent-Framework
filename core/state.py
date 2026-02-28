import uuid

class ResearchState:
    def __init__(self, goal):
        self.run_id = str(uuid.uuid4())
        self.goal = goal
        self.search_count = 0
        self.iterations = 0
        self.research_items = []
        self.logs = []
        self.finished = False
        self.total_tokens = 0
        self.estimated_cost = 0
        self.query_history = set()

    def add_research_item(self, item):
        self.research_items.append(item)

    def log(self, message):
        self.logs.append(message)

    def add_query(self, query):
        self.query_history.add(query)