import wolframalpha

# Use your own Wolfram Alpha app id
app_id = "8TJ67Y-PJ8R4338TQ"
client = wolframalpha.Client(app_id)

# Enter the quadratic equation
# request = "Find the roots of x^2 - 2*x + 5 = 0"
# request = "Plot y=x^2"
# request = "temperature plot for London from 2023-05-26 to 2023-06-05"
request = "weather tomorrow"

resp = client.query(request)

print(resp)

# The Result of query is a list of pods
# The solution is usually in the 'Result' pod, hence we will print the text of the 'Result' pod
for pod in resp.pods:
    if pod.title == "Result":
        print(pod.text)
    elif pod.title == "Results":
        for idx, sub in enumerate(pod.subpods):
            print(f"Result[{idx}]: {sub.plaintext}")
    elif pod.title == "Plot":
        for sub in pod.subpods:
            print("Plot: " + sub.img.src)
