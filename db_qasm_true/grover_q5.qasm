OPENQASM 2.0;
include "qelib1.inc";
gate mcphase(param0) q0,q1,q2,q3,q4 { u(pi/2,0,pi) q4; cx q1,q4; p(-pi/4) q4; cx q0,q4; p(pi/4) q4; cx q1,q4; p(pi/4) q1; p(-pi/4) q4; cx q0,q4; cx q0,q1; p(pi/4) q0; p(-pi/4) q1; cx q0,q1; u(pi/4,pi/2,-pi/4) q4; cx q3,q4; p(-pi/4) q4; cx q2,q4; p(pi/4) q4; cx q3,q4; p(pi/4) q3; p(-pi/4) q4; cx q2,q4; cx q2,q3; p(pi/4) q2; p(-pi/4) q3; cx q2,q3; u(pi/4,-pi/2,3*pi/4) q4; cx q1,q4; p(-pi/4) q4; cx q0,q4; p(pi/4) q4; cx q1,q4; p(pi/4) q1; p(-pi/4) q4; cx q0,q4; cx q0,q1; p(pi/4) q0; p(-pi/4) q1; cx q0,q1; u(pi/4,pi/2,-pi/4) q4; cx q3,q4; p(-pi/4) q4; cx q2,q4; p(pi/4) q4; cx q3,q4; p(pi/4) q3; p(-pi/4) q4; cx q2,q4; cx q2,q3; p(pi/4) q2; p(-pi/4) q3; cx q2,q3; u(pi/2,pi/4,-3*pi/4) q4; u(pi/2,0,pi) q3; cx q1,q3; p(-pi/4) q3; cx q0,q3; p(pi/4) q3; cx q1,q3; p(pi/4) q1; p(-pi/4) q3; cx q0,q3; cx q0,q1; p(pi/4) q0; p(-pi/4) q1; cx q0,q1; u(pi/2,-pi/8,-3*pi/4) q3; cx q2,q3; u(pi/2,0,-7*pi/8) q3; cx q1,q3; p(-pi/4) q3; cx q0,q3; p(pi/4) q3; cx q1,q3; p(pi/4) q1; p(-pi/4) q3; cx q0,q3; cx q0,q1; p(pi/4) q0; p(-pi/4) q1; cx q0,q1; u(pi/2,-pi/8,-3*pi/4) q3; cx q2,q3; u(0,-15*pi/16,-15*pi/16) q3; cx q0,q2; u(0,-pi/32,-pi/32) q2; cx q1,q2; u(0,-3.043417883165112,-3.0434178831651124) q2; cx q0,q2; u(0,-pi/32,-pi/32) q2; cx q1,q2; u(0,-3.043417883165112,-3.0434178831651124) q2; u(0,0,pi/16) q1; cx q0,q1; u(0,0,-pi/16) q1; cx q0,q1; p(pi/16) q0; }
gate mcx q0,q1,q2,q3 { h q3; p(pi/8) q0; p(pi/8) q1; p(pi/8) q2; p(pi/8) q3; cx q0,q1; p(-pi/8) q1; cx q0,q1; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; h q3; }
gate mcx_2004276383328 q0,q1,q2,q3 { mcx q0,q1,q2,q3; }
gate gate_Q q0,q1,q2,q3,q4 { mcphase(pi) q0,q1,q2,q3,q4; h q3; h q2; h q1; h q0; x q0; x q1; x q2; x q3; h q3; mcx_2004276383328 q0,q1,q2,q3; h q3; x q0; x q1; x q2; x q3; h q0; h q1; h q2; h q3; }
gate gate_Q_2004274936192 q0,q1,q2,q3,q4 { gate_Q q0,q1,q2,q3,q4; }
gate gate_Q_2004276386832 q0,q1,q2,q3,q4 { gate_Q q0,q1,q2,q3,q4; }
gate gate_Q_2004275575216 q0,q1,q2,q3,q4 { gate_Q q0,q1,q2,q3,q4; }
qreg q[4];
qreg flag[1];
creg meas[5];
h q[0];
h q[1];
h q[2];
h q[3];
x flag[0];
gate_Q_2004274936192 q[0],q[1],q[2],q[3],flag[0];
gate_Q_2004276386832 q[0],q[1],q[2],q[3],flag[0];
gate_Q_2004275575216 q[0],q[1],q[2],q[3],flag[0];
barrier q[0],q[1],q[2],q[3],flag[0];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure flag[0] -> meas[4];