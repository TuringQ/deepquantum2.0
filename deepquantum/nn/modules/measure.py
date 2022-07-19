import torch
from deepquantum.gates.qoperator import PauliX, PauliY, PauliZ


def measure_mps(state, n_qubits, pauli_string=None, dev='cpu'):
    # state: MPS
    if dev == "gpu" or dev == "cuda":
        assert state.is_cuda, "------state must be on-cuda-----"

    assert state.ndim == n_qubits, "ndim of state must be n_qubits"
    for i in range(n_qubits):
        assert state.shape[i] == 2, "ndim of shape i of state must be 2"

    if pauli_string != None:
        pauli_lst = pauli_string.split(',')
        pauli_lst = [i.strip() for i in pauli_lst]
        pauli_lst = [(i[0], int(i[1:])) for i in pauli_lst]
    else:
        pauli_lst = [('z', i) for i in range(n_qubits)]

    device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
    measures = torch.zeros(len(pauli_lst), dtype=torch.float32).to(device)
    for i, p in enumerate(pauli_lst):
        mps = torch.clone(state)
        pauli, qubit = p
        if (qubit >= n_qubits) or (qubit < 0):
            raise ValueError('qubit must less than n_qubits')
        if pauli.lower() == 'x':
            pGate = PauliX(n_qubits, qubit, dev)
        elif pauli.lower() == 'y':
            pGate = PauliY(n_qubits, qubit, dev)
        elif pauli.lower() == 'z':
            pGate = PauliZ(n_qubits, qubit, dev)
        mps = pGate.TN_contract(mps)
        res = (state.reshape(1, -1).conj()) @ (mps.reshape(-1, 1))
        measures[i] = res.squeeze().real
    return measures


def measure_mps_batch(state, n_qubits, batch_size, pauli_string=None, dev='cpu'):
    if dev == "gpu" or dev == "cuda":
        assert state.is_cuda, "------state must be on-cuda-----"

    assert state.ndim == n_qubits + 1, "ndim of input must be n_qubits+1"
    assert state.shape[0] == batch_size, "shape[0] of state must be batch size"
    for i in range(1, n_qubits+1):
        assert state.shape[i] == 2, "ndim of component shape of input must be 2"
    if pauli_string != None:
        pauli_lst = pauli_string.split(',')
        pauli_lst = [i.strip() for i in pauli_lst]
        pauli_lst = [(i[0], int(i[1:])) for i in pauli_lst]
    else:
        pauli_lst = [('z', i) for i in range(n_qubits)]

    device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
    measures = torch.zeros(len(pauli_lst), batch_size, dtype=torch.float32).to(device)
    for i, p in enumerate(pauli_lst):
        mps = torch.clone(state)
        pauli, qubit = p
        if (qubit >= n_qubits) or (qubit < 0):
            raise ValueError('qubit must less than n_qubits')
        if pauli.lower() == 'x':
            p_gate = PauliX(n_qubits, qubit, dev)
        elif pauli.lower() == 'y':
            p_gate = PauliY(n_qubits, qubit, dev)
        elif pauli.lower() == 'z':
            p_gate = PauliZ(n_qubits, qubit, dev)
        mps = p_gate.TN_contract(mps, batch_mod=True)
        res = (state.reshape(batch_size, 1, -1).conj()) @ (mps.reshape(batch_size, -1, 1))
        measures[i] = res.squeeze().real
    return measures.permute(1, 0)


def measure_rho(dm, n_qubits, pauli_string=None, dev='cpu'):
    if dev == "gpu" or dev == "cuda":
        assert dm.is_cuda, "------dm must be on-cuda-----"

    assert (dm.shape[0] == (1 << n_qubits)) and (dm.shape[1] == (1 << n_qubits)), "input must be 2**n-by-2**n"
    if pauli_string != None:
        pauli_lst = pauli_string.split(',')
        pauli_lst = [i.strip() for i in pauli_lst]
        pauli_lst = [(i[0], int(i[1:])) for i in pauli_lst]
    else:
        pauli_lst = [('z', i) for i in range(n_qubits)]

    device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
    measures = torch.zeros(len(pauli_lst), dtype=torch.float32).to(device)
    for i, p in enumerate(pauli_lst):
        pauli, qubit = p
        if qubit >= n_qubits:
            raise ValueError('qubit must less than n_qubits')
        if pauli.lower() == 'x':
            p_gate = PauliX(n_qubits, qubit, dev)
        elif pauli.lower() == 'y':
            p_gate = PauliY(n_qubits, qubit, dev)
        elif pauli.lower() == 'z':
            p_gate = PauliZ(n_qubits, qubit, dev)
        mat = p_gate.U_expand()
        d_row = dm.reshape(1, -1)
        m_col = mat.permute(1, 0).reshape(-1, 1)
        res = d_row @ m_col
        measures[i] = res.squeeze().real
    return measures


def measure_rho_batch(dm, n_qubits, batch_size, pauli_string=None, dev='cpu'):
    if dev == "gpu" or dev == "cuda":
        assert dm.is_cuda, "------dm must be on-cuda-----"

    assert (dm.shape[0] == batch_size) and (dm.shape[1] == (1 << n_qubits)) and (dm.shape[2] == (1 << n_qubits)),\
        "input must be batch*2**n*2**n"
    if pauli_string != None:
        pauli_lst = pauli_string.split(',')
        pauli_lst = [i.strip() for i in pauli_lst]
        pauli_lst = [(i[0], int(i[1:])) for i in pauli_lst]
    else:
        pauli_lst = [('z', i) for i in range(n_qubits)]

    device = torch.device("cuda" if (dev == "gpu" or dev == "cuda") else "cpu")
    measures = torch.zeros(len(pauli_lst), batch_size, dtype=torch.float32).to(device)
    for i, p in enumerate(pauli_lst):
        pauli, qubit = p
        if qubit >= n_qubits:
            raise ValueError('qubit must less than n_qubits')
        if pauli.lower() == 'x':
            p_gate = PauliX(n_qubits, qubit, dev)
        elif pauli.lower() == 'y':
            p_gate = PauliY(n_qubits, qubit, dev)
        elif pauli.lower() == 'z':
            p_gate = PauliZ(n_qubits, qubit, dev)
        mat = p_gate.U_expand()
        d_row = dm.reshape(batch_size, 1, -1)
        m_col = mat.permute(1, 0).reshape(-1, 1)
        res = d_row @ m_col
        measures[i] = res.squeeze().real
    return measures.permute(1, 0)



def measure_mps_batch_test():
    n_qubits = 8
    batch_size = 5
    x = torch.nn.functional.normalize(torch.rand(batch_size, 1, 2 ** n_qubits) + 0j, p=2, dim=2)
    print(f'x.shape: {x.shape}')
    x = x.reshape([batch_size] + [2] * n_qubits)
    res = measure_mps_batch(x, n_qubits, batch_size, 'x0, y1,z2')
    print(f'res.shape: {res.shape}')


def measure_mps_batch_cuda_test():
    n_qubits = 8
    batch_size = 5
    x = torch.nn.functional.normalize(torch.rand(batch_size, 1, 2 ** n_qubits).cuda() + 0j, p=2, dim=2)
    print(f'x.shape: {x.shape}')
    x = x.reshape([batch_size] + [2] * n_qubits)
    res = measure_mps_batch(x, n_qubits, batch_size, 'x0, y1,z2', dev='cuda')
    print(f'res.shape: {res.shape}')
    print(f'res.device: {res.device}')


def measure_rho_test():
    n_qubits = 8
    x = torch.nn.functional.normalize(torch.rand(1, 2 ** n_qubits) + torch.rand(1, 2 ** n_qubits) * 1j, p=2, dim=1)
    dm = x.reshape(-1, 1) @ x.conj()
    res = measure_rho(dm, n_qubits, 'x0,y2, z3')
    print(res.shape)


def measure_rho_cuda_test():
    n_qubits = 8
    x = torch.nn.functional.normalize(torch.rand(1, 2 ** n_qubits).cuda() + torch.rand(1, 2 ** n_qubits).cuda() * 1j, p=2, dim=1)
    dm = x.reshape(-1, 1) @ x.conj()
    res = measure_rho(dm, n_qubits, 'x0,y2, z3', dev='cuda')
    print(res.shape)
    print(f'res.device: {res.device}')



def measure_rho_batch_test():
    n_qubits = 8
    batch_size = 4
    x = torch.nn.functional.normalize(torch.rand(batch_size, 1, 2 ** n_qubits) +
                                      torch.rand(batch_size, 1, 2 ** n_qubits) * 1j, p=2, dim=2)
    dm = x.reshape(batch_size, -1, 1) @ x.conj()
    print(dm.shape)
    res = measure_rho_batch(dm, n_qubits, batch_size,'x0,y2, z3')
    print(res.shape)


def measure_rho_batch_cuda_test():
    n_qubits = 8
    batch_size = 4
    x = torch.nn.functional.normalize(torch.rand(batch_size, 1, 2 ** n_qubits).cuda() +
                                      torch.rand(batch_size, 1, 2 ** n_qubits).cuda() * 1j, p=2, dim=2)
    dm = x.reshape(batch_size, -1, 1) @ x.conj()
    print(dm.shape)
    res = measure_rho_batch(dm, n_qubits, batch_size,'x0,y2, z3', dev='cuda')
    print(res.shape)
    print(f'res.device: {res.device}')



if __name__ == "__main__":
    # measure_mps_batch_test()
    measure_mps_batch_cuda_test()

    # measure_rho_test()
    # measure_rho_cuda_test()

    # measure_rho_batch_test()
    # measure_rho_batch_cuda_test()
