import asyncio
import nodes.diffusion.clip_text_encode as node

print("Loaded from:", node.__file__)
print("Declared outputs:", [o["var_name"] for o in node.outputs])

async def main():
    result = await node.execute("test_uid", "test_token", {"prompt": "a photo of a cat", "negative_prompt": ""})
    print("Keys:", list(result.keys()))
    print("shape_debug:", result.get("shape_debug"))

asyncio.run(main())
