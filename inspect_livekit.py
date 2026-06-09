from livekit.agents import AgentSession

print("Methods containing 'transcribed':")
for m in dir(AgentSession):
    if "transcribed" in m.lower():
        print(m)

print("\nMethods containing 'input':")
for m in dir(AgentSession):
    if "input" in m.lower():
        print(m)