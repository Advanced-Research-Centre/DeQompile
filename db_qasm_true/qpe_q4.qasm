OPENQASM 2.0;
include "qelib1.inc";
gate gate_IQFT_dg q0,q1,q2 { swap q0,q2; h q0; cp(-pi/2) q1,q0; h q1; cp(-pi/4) q2,q0; cp(-pi/2) q2,q1; h q2; }
qreg q[3];
qreg psi[1];
creg c[3];
x psi[0];
h q[0];
h q[1];
h q[2];
cp(pi) psi[0],q[0];
gate_IQFT_dg q[0],q[1],q[2];
barrier q[0],q[1],q[2],psi[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];