from nodes.registry import registry

print("Total nodes found:", len(registry))
for key in registry:
    print(" -", key)