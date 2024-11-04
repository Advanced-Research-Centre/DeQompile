OPENQASM 2.0;
include "qelib1.inc";
gate gate_QFT q0,q1,q2,q3 { h q3; cp(pi/2) q3,q2; cp(pi/4) q3,q1; cp(pi/8) q3,q0; h q2; cp(pi/2) q2,q1; cp(pi/4) q2,q0; h q1; cp(pi/2) q1,q0; h q0; swap q0,q3; swap q1,q2; }
qreg q[4];
creg c[4];
creg meas[4];
gate_QFT q[0],q[1],q[2],q[3];
barrier q[0],q[1],q[2],q[3];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];