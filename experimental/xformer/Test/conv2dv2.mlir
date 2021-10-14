// RUN: xcore-opt --mlir-io %s --xcore-replace-with-conv2dv2 | FileCheck %s

// CHECK-LABEL: padded_indirect
func @padded_indirect(%arg0: tensor<?x1x1x4x!quant.uniform<i8:f32, 0.0077245929278433323:-1>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 0.010062812827527523:-18>> attributes {tf.entry_function = {inputs = "conv2d_input", outputs = "Identity"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<4x1x9x4x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0078216465190052986,0.0078239291906356812,0.0078047262504696846}>>, value = dense<1> : tensor<4x1x9x4xi8>} : () -> tensor<4x1x9x4x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0078216465190052986,0.0078239291906356812,0.0078047262504696846}>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<4x!quant.uniform<i32:f32:0, {6.0732123529305682E-5,6.0419035435188562E-5,6.0436668718466535E-5,6.0288333770586178E-5}>>, value = dense<0> : tensor<4xi32>} : () -> tensor<4x!quant.uniform<i32:f32:0, {6.0732123529305682E-5,6.0419035435188562E-5,6.0436668718466535E-5,6.0288333770586178E-5}>>
  
  // CHECK: xc.conv2d_v2
  // CHECK-SAME: PaddedIndirect
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x1x1x4x!quant.uniform<i8:f32, 0.0077245929278433323:-1>>, tensor<4x1x9x4x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0078216465190052986,0.0078239291906356812,0.0078047262504696846}>>, tensor<4x!quant.uniform<i32:f32:0, {6.0732123529305682E-5,6.0419035435188562E-5,6.0436668718466535E-5,6.0288333770586178E-5}>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 0.010062812827527523:-18>>
  return %2 : tensor<?x1x1x4x!quant.uniform<i8:f32, 0.010062812827527523:-18>>
}

// -----

// CHECK-LABEL: valid_indirect
func @valid_indirect(%arg0: tensor<?x4x4x16x!quant.uniform<i8:f32, 0.0078378040343523026:-1>>) -> tensor<?x2x2x16x!quant.uniform<i8:f32, 0.1095665842294693:9>> attributes {tf.entry_function = {inputs = "conv2d_input", outputs = "Identity"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x3x3x16x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0077851288951933384,0.0076941768638789654,0.007869236171245575,0.007815798744559288,0.0078112063929438591,0.0077464901842176914,0.0078260786831378936,0.0078160781413316727,0.0078216465190052986,0.0078375879675149918,0.0078175142407417297,0.0078369667753577232,0.0078147416934370995,0.0078694447875022888,0.0078470427542924881}>>, value = dense<1> : tensor<16x3x3x16xi8>} : () -> tensor<16x3x3x16x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0077851288951933384,0.0076941768638789654,0.007869236171245575,0.007815798744559288,0.0078112063929438591,0.0077464901842176914,0.0078260786831378936,0.0078160781413316727,0.0078216465190052986,0.0078375879675149918,0.0078175142407417297,0.0078369667753577232,0.0078147416934370995,0.0078694447875022888,0.0078470427542924881}>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32:0, {6.1622209614142776E-5,6.1018316046101972E-5,6.0305450460873544E-5,6.1677528719883412E-5,6.1258695495780557E-5,6.1222708609420806E-5,6.0715472500305623E-5,6.1339269450400025E-5,6.1260885559022427E-5,6.1304534028749913E-5,6.1429476772900671E-5,6.1272141465451568E-5,6.1424609157256782E-5,6.1250415456015617E-5,6.1679165810346603E-5,6.1503582401201129E-5}>>, value = dense<0> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32:0, {6.1622209614142776E-5,6.1018316046101972E-5,6.0305450460873544E-5,6.1677528719883412E-5,6.1258695495780557E-5,6.1222708609420806E-5,6.0715472500305623E-5,6.1339269450400025E-5,6.1260885559022427E-5,6.1304534028749913E-5,6.1429476772900671E-5,6.1272141465451568E-5,6.1424609157256782E-5,6.1250415456015617E-5,6.1679165810346603E-5,6.1503582401201129E-5}>>

  // CHECK: xc.conv2d_v2
  // CHECK-SAME: ValidIndirect
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x4x4x16x!quant.uniform<i8:f32, 0.0078378040343523026:-1>>, tensor<16x3x3x16x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0077851288951933384,0.0076941768638789654,0.007869236171245575,0.007815798744559288,0.0078112063929438591,0.0077464901842176914,0.0078260786831378936,0.0078160781413316727,0.0078216465190052986,0.0078375879675149918,0.0078175142407417297,0.0078369667753577232,0.0078147416934370995,0.0078694447875022888,0.0078470427542924881}>>, tensor<16x!quant.uniform<i32:f32:0, {6.1622209614142776E-5,6.1018316046101972E-5,6.0305450460873544E-5,6.1677528719883412E-5,6.1258695495780557E-5,6.1222708609420806E-5,6.0715472500305623E-5,6.1339269450400025E-5,6.1260885559022427E-5,6.1304534028749913E-5,6.1429476772900671E-5,6.1272141465451568E-5,6.1424609157256782E-5,6.1250415456015617E-5,6.1679165810346603E-5,6.1503582401201129E-5}>>) -> tensor<?x2x2x16x!quant.uniform<i8:f32, 0.1095665842294693:9>>
  return %2 : tensor<?x2x2x16x!quant.uniform<i8:f32, 0.1095665842294693:9>>
}

// -----

// CHECK-LABEL: valid_direct1
func @valid_direct1(%arg0: tensor<?x4x4x32x!quant.uniform<i8:f32, 0.0078387223184108734:-1>>) -> tensor<?x2x2x16x!quant.uniform<i8:f32, 0.11676134914159775:10>> attributes {tf.entry_function = {inputs = "conv2d_input", outputs = "Identity"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x3x3x32x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0078625492751598358,0.0078548826277256011,0.007869236171245575,0.0078516323119401932,0.0078112063929438591,0.0078599303960800171,0.0078289676457643509,0.0078668594360351563,0.0078529221937060356,0.0078452629968523979,0.0078603299334645271,0.0078511722385883331,0.0078590558841824532,0.0078701050952076912,0.0078737959265708923}>>, value = dense<1> : tensor<16x3x3x32xi8>} : () -> tensor<16x3x3x32x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0078625492751598358,0.0078548826277256011,0.007869236171245575,0.0078516323119401932,0.0078112063929438591,0.0078599303960800171,0.0078289676457643509,0.0078668594360351563,0.0078529221937060356,0.0078452629968523979,0.0078603299334645271,0.0078511722385883331,0.0078590558841824532,0.0078701050952076912,0.0078737959265708923}>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32:0, {6.1629427364096045E-5,6.1632337747141719E-5,6.1572245613206178E-5,6.1684753745794296E-5,6.1546765209641308E-5,6.1229875427670777E-5,6.1611812270712107E-5,6.1369100876618177E-5,6.1666127294301987E-5,6.1556878790725023E-5,6.1496837588492781E-5,6.1614940932486206E-5,6.1543156334664673E-5,6.1604958318639547E-5,6.1691571318078786E-5,6.1720500525552779E-5}>>, value = dense<0> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32:0, {6.1629427364096045E-5,6.1632337747141719E-5,6.1572245613206178E-5,6.1684753745794296E-5,6.1546765209641308E-5,6.1229875427670777E-5,6.1611812270712107E-5,6.1369100876618177E-5,6.1666127294301987E-5,6.1556878790725023E-5,6.1496837588492781E-5,6.1614940932486206E-5,6.1543156334664673E-5,6.1604958318639547E-5,6.1691571318078786E-5,6.1720500525552779E-5}>>
  
  // CHECK: xc.conv2d_v2
  // CHECK-SAME: ValidDirect
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x4x4x32x!quant.uniform<i8:f32, 0.0078387223184108734:-1>>, tensor<16x3x3x32x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0078625492751598358,0.0078548826277256011,0.007869236171245575,0.0078516323119401932,0.0078112063929438591,0.0078599303960800171,0.0078289676457643509,0.0078668594360351563,0.0078529221937060356,0.0078452629968523979,0.0078603299334645271,0.0078511722385883331,0.0078590558841824532,0.0078701050952076912,0.0078737959265708923}>>, tensor<16x!quant.uniform<i32:f32:0, {6.1629427364096045E-5,6.1632337747141719E-5,6.1572245613206178E-5,6.1684753745794296E-5,6.1546765209641308E-5,6.1229875427670777E-5,6.1611812270712107E-5,6.1369100876618177E-5,6.1666127294301987E-5,6.1556878790725023E-5,6.1496837588492781E-5,6.1614940932486206E-5,6.1543156334664673E-5,6.1604958318639547E-5,6.1691571318078786E-5,6.1720500525552779E-5}>>) -> tensor<?x2x2x16x!quant.uniform<i8:f32, 0.11676134914159775:10>>
  return %2 : tensor<?x2x2x16x!quant.uniform<i8:f32, 0.11676134914159775:10>>
}

// -----

// CHECK-LABEL: valid_direct2
func @valid_direct2(%arg0: tensor<?x4x4x32x!quant.uniform<i8:f32, 0.0078387223184108734:-1>>) -> tensor<?x4x4x16x!quant.uniform<i8:f32, 0.054258987307548523:12>> attributes {tf.entry_function = {inputs = "conv2d_input", outputs = "Identity"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x1x1x32x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0075159505940973759,0.0076126507483422756,0.0073894253000617027,0.0068393861874938011,0.0077006686478853226,0.0075865150429308414,0.0078260786831378936,0.0077599729411303997,0.0078216465190052986,0.0078239291906356812,0.0078081502579152584,0.0078369667753577232,0.0077880122698843479,0.0077783181332051754,0.0078047262504696846}>>, value = dense<1> : tensor<16x1x1x32xi8>} : () -> tensor<16x1x1x32x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0075159505940973759,0.0076126507483422756,0.0073894253000617027,0.0068393861874938011,0.0077006686478853226,0.0075865150429308414,0.0078260786831378936,0.0077599729411303997,0.0078216465190052986,0.0078239291906356812,0.0078081502579152584,0.0078369667753577232,0.0077880122698843479,0.0077783181332051754,0.0078047262504696846}>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32:0, {6.1629427364096045E-5,5.8915447880281135E-5,5.9673453506547958E-5,5.7923654821934178E-5,5.3612049669027328E-5,6.0363403463270515E-5,5.9468584368005395E-5,6.1346458096522838E-5,6.0828271671198308E-5,6.1311715398915112E-5,6.1329606978688389E-5,6.1205922975204885E-5,6.1431805079337209E-5,6.1048063798807561E-5,6.0972077335463837E-5,6.1179081967566162E-5}>>, value = dense<0> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32:0, {6.1629427364096045E-5,5.8915447880281135E-5,5.9673453506547958E-5,5.7923654821934178E-5,5.3612049669027328E-5,6.0363403463270515E-5,5.9468584368005395E-5,6.1346458096522838E-5,6.0828271671198308E-5,6.1311715398915112E-5,6.1329606978688389E-5,6.1205922975204885E-5,6.1431805079337209E-5,6.1048063798807561E-5,6.0972077335463837E-5,6.1179081967566162E-5}>>
  
  // CHECK: xc.conv2d_v2
  // CHECK-SAME: ValidDirect
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x4x4x32x!quant.uniform<i8:f32, 0.0078387223184108734:-1>>, tensor<16x1x1x32x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0075159505940973759,0.0076126507483422756,0.0073894253000617027,0.0068393861874938011,0.0077006686478853226,0.0075865150429308414,0.0078260786831378936,0.0077599729411303997,0.0078216465190052986,0.0078239291906356812,0.0078081502579152584,0.0078369667753577232,0.0077880122698843479,0.0077783181332051754,0.0078047262504696846}>>, tensor<16x!quant.uniform<i32:f32:0, {6.1629427364096045E-5,5.8915447880281135E-5,5.9673453506547958E-5,5.7923654821934178E-5,5.3612049669027328E-5,6.0363403463270515E-5,5.9468584368005395E-5,6.1346458096522838E-5,6.0828271671198308E-5,6.1311715398915112E-5,6.1329606978688389E-5,6.1205922975204885E-5,6.1431805079337209E-5,6.1048063798807561E-5,6.0972077335463837E-5,6.1179081967566162E-5}>>) -> tensor<?x4x4x16x!quant.uniform<i8:f32, 0.054258987307548523:12>>
  return %2 : tensor<?x4x4x16x!quant.uniform<i8:f32, 0.054258987307548523:12>>
}

// -----

// CHECK-LABEL: invalid_output_depth
func @invalid_output_depth(%arg0: tensor<?x1x1x4x!quant.uniform<i8:f32, 0.0077245929278433323:-1>>) -> tensor<?x1x1x3x!quant.uniform<i8:f32, 0.013466199859976768:31>> attributes {tf.entry_function = {inputs = "conv2d_input", outputs = "Identity"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<3x1x9x4x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0078216465190052986,0.0078047262504696846}>>, value = dense<1> : tensor<3x1x9x4xi8>} : () -> tensor<3x1x9x4x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0078216465190052986,0.0078047262504696846}>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<3x!quant.uniform<i32:f32:0, {6.0732123529305682E-5,6.0419035435188562E-5,6.0288333770586178E-5}>>, value = dense<0> : tensor<3xi32>} : () -> tensor<3x!quant.uniform<i32:f32:0, {6.0732123529305682E-5,6.0419035435188562E-5,6.0288333770586178E-5}>>

  // CHECK-NOT: xc.conv2d_v2
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x1x1x4x!quant.uniform<i8:f32, 0.0077245929278433323:-1>>, tensor<3x1x9x4x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0078216465190052986,0.0078047262504696846}>>, tensor<3x!quant.uniform<i32:f32:0, {6.0732123529305682E-5,6.0419035435188562E-5,6.0288333770586178E-5}>>) -> tensor<?x1x1x3x!quant.uniform<i8:f32, 0.013466199859976768:31>>
  return %2 : tensor<?x1x1x3x!quant.uniform<i8:f32, 0.013466199859976768:31>>
}

// -----

// CHECK-LABEL: invalid_input_depth
func @invalid_input_depth(%arg0: tensor<?x1x1x3x!quant.uniform<i8:f32, 0.0076398402452468872:1>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 0.0078214062377810478:16>> attributes {tf.entry_function = {inputs = "conv2d_input", outputs = "Identity"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<4x1x9x3x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0078216465190052986,0.0078239291906356812,0.0078047262504696846}>>, value = dense<1> : tensor<4x1x9x3xi8>} : () -> tensor<4x1x9x3x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0078216465190052986,0.0078239291906356812,0.0078047262504696846}>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<4x!quant.uniform<i32:f32:0, {6.0065780417062342E-5,5.9756130212917924E-5,5.9773570683319122E-5,5.9626861911965534E-5}>>, value = dense<0> : tensor<4xi32>} : () -> tensor<4x!quant.uniform<i32:f32:0, {6.0065780417062342E-5,5.9756130212917924E-5,5.9773570683319122E-5,5.9626861911965534E-5}>>
  
  // CHECK-NOT: xc.conv2d_v2
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x1x1x3x!quant.uniform<i8:f32, 0.0076398402452468872:1>>, tensor<4x1x9x3x!quant.uniform<i8<-127:127>:f32:0, {0.0078621776774525642,0.0078216465190052986,0.0078239291906356812,0.0078047262504696846}>>, tensor<4x!quant.uniform<i32:f32:0, {6.0065780417062342E-5,5.9756130212917924E-5,5.9773570683319122E-5,5.9626861911965534E-5}>>) -> tensor<?x1x1x4x!quant.uniform<i8:f32, 0.0078214062377810478:16>>
  return %2 : tensor<?x1x1x4x!quant.uniform<i8:f32, 0.0078214062377810478:16>>
}