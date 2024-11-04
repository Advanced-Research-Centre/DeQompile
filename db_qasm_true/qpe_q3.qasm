OPENQASM 2.0;
include "qelib1.inc";
gate gate_IQFT_dg q0,q1 { swap q0,q1; h q0; cp(-pi/2) q1,q0; h q1; }
qreg q[2];
qreg psi[1];
creg c[2];
x psi[0];
h q[0];
h q[1];
cp(pi) psi[0],q[0];
gate_IQFT_dg q[0],q[1];
barrier q[0],q[1],psi[0];
measure q[0] -> c[0];
measure q[1] -> c[1];