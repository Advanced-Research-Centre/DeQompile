OPENQASM 2.0;
include "qelib1.inc";
gate gate_IQFT_dg q0 { h q0; }
qreg q[1];
qreg psi[1];
creg c[1];
x psi[0];
h q[0];
cp(pi) psi[0],q[0];
gate_IQFT_dg q[0];
barrier q[0],psi[0];
measure q[0] -> c[0];