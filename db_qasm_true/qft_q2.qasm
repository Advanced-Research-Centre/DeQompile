OPENQASM 2.0;
include "qelib1.inc";
gate gate_QFT q0,q1 { h q1; cp(pi/2) q1,q0; h q0; swap q0,q1; }
qreg q[2];
creg c[2];
creg meas[2];
gate_QFT q[0],q[1];
barrier q[0],q[1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];