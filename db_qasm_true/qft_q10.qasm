OPENQASM 2.0;
include "qelib1.inc";
gate gate_QFT q0,q1,q2,q3,q4,q5,q6,q7,q8,q9 { h q9; cp(pi/2) q9,q8; cp(pi/4) q9,q7; cp(pi/8) q9,q6; cp(pi/16) q9,q5; cp(pi/32) q9,q4; cp(pi/64) q9,q3; cp(pi/128) q9,q2; cp(pi/256) q9,q1; cp(pi/512) q9,q0; h q8; cp(pi/2) q8,q7; cp(pi/4) q8,q6; cp(pi/8) q8,q5; cp(pi/16) q8,q4; cp(pi/32) q8,q3; cp(pi/64) q8,q2; cp(pi/128) q8,q1; cp(pi/256) q8,q0; h q7; cp(pi/2) q7,q6; cp(pi/4) q7,q5; cp(pi/8) q7,q4; cp(pi/16) q7,q3; cp(pi/32) q7,q2; cp(pi/64) q7,q1; cp(pi/128) q7,q0; h q6; cp(pi/2) q6,q5; cp(pi/4) q6,q4; cp(pi/8) q6,q3; cp(pi/16) q6,q2; cp(pi/32) q6,q1; cp(pi/64) q6,q0; h q5; cp(pi/2) q5,q4; cp(pi/4) q5,q3; cp(pi/8) q5,q2; cp(pi/16) q5,q1; cp(pi/32) q5,q0; h q4; cp(pi/2) q4,q3; cp(pi/4) q4,q2; cp(pi/8) q4,q1; cp(pi/16) q4,q0; h q3; cp(pi/2) q3,q2; cp(pi/4) q3,q1; cp(pi/8) q3,q0; h q2; cp(pi/2) q2,q1; cp(pi/4) q2,q0; h q1; cp(pi/2) q1,q0; h q0; swap q0,q9; swap q1,q8; swap q2,q7; swap q3,q6; swap q4,q5; }
qreg q[10];
creg c[10];
creg meas[10];
gate_QFT q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];