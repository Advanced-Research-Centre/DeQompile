OPENQASM 2.0;
include "qelib1.inc";
gate mcphase(param0) q0,q1,q2 { cx q0,q2; u(0,-pi/8,-pi/8) q2; cx q1,q2; u(0,-7*pi/8,-7*pi/8) q2; cx q0,q2; u(0,-pi/8,-pi/8) q2; cx q1,q2; u(0,-7*pi/8,-7*pi/8) q2; u(0,0,pi/4) q1; cx q0,q1; u(0,0,-pi/4) q1; cx q0,q1; p(pi/4) q0; }
gate gate_Q q0,q1,q2 { mcphase(pi) q0,q1,q2; h q1; h q0; x q0; x q1; h q1; cx q0,q1; h q1; x q0; x q1; h q0; h q1; }
gate gate_Q_2004273462400 q0,q1,q2 { gate_Q q0,q1,q2; }
qreg q[2];
qreg flag[1];
creg meas[3];
h q[0];
h q[1];
x flag[0];
gate_Q_2004273462400 q[0],q[1],flag[0];
barrier q[0],q[1],flag[0];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure flag[0] -> meas[2];