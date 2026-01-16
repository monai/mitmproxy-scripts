# mitmproxy-scripts

Install mitmproxy with the required dependencies:

```shell
uv tool install --with "Jinja2,openai,pydantic,pyyaml" mitmproxy
```

Run:

```shell
mitmweb --mode reverse:http://127.0.0.1:8023 --listen-port 8013 --script scripts/llama_cpp.py
```
