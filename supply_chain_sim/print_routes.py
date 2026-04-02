import json
from server.app import app
print(json.dumps(app.openapi(), indent=2))
