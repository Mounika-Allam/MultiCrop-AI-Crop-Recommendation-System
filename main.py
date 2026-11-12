import os
from app import app

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    print(f"Starting server on http://{host}:{port} with debug={debug}...")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
