from qiskit_ibm_runtime import QiskitRuntimeService

# The error is persisting because your local machine's hidden Qiskit credentials 
# file (~/.qiskit/qiskit-ibm.json) is still caching the dead URL. 

# Run this snippet exactly once to forcefully overwrite the corrupted cache.
QiskitRuntimeService.save_account(
    channel='ibm_quantum_platform',
    token="K7yJgf-a4CR420h0VbpOMRXURc1XUrQp2SWjCS2ZGz7H",
    overwrite=True,
    set_as_default=True
)

service = QiskitRuntimeService(channel='ibm_quantum_platform')
print("Successfully connected to:", service.active_account()['url'])