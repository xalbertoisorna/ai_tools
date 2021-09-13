// RUN: xcore-opt --mlir-io %s --xcore-apply-patterns --xcore-legalize-fullyconnected | FileCheck %s

// CHECK-LABEL: check_bias_transformation
func @check_bias_transformation(%arg0: tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>> attributes {tf.entry_function = {inputs = "flatten_input_int8", outputs = "Identity_int8"}} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, value = dense<1> : tensor<32x32xi8>} : () -> tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>
// CHECK: "0x000000000000000000000000000000000000000000000000000000000000000001000100010001000100010001000100010001000100010001000100010001000300030003000300030003000300030003000300030003000300030003000300FB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6B65146514651465146514651465146514651465146514651465146514651465149AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB1500150015001500150015001500150015001500150015001500150015001500000000000000000000000000000000000000000000000000000000000000000001000100010001000100010001000100010001000100010001000100010001000300030003000300030003000300030003000300030003000300030003000300FB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6BFB6B65146514651465146514651465146514651465146514651465146514651465149AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB9AEB1500150015001500150015001500150015001500150015001500150015001500"
// CHECK-SAME: tensor<2x7x16xi16>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>, value = dense<1> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>
// CHECK: xc.fc
  %2 = "tfl.fully_connected"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x4x8x1x!quant.uniform<i8:f32, 0.0078160231932997704>>, tensor<32x32x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, tensor<32x!quant.uniform<i32:f32, 6.1507766076829284E-5>>) -> tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
// CHECK: return
  return %2 : tensor<1x32x!quant.uniform<i8:f32, 0.037329975515604019:-13>>
}

// -----

// CHECK-LABEL: check_weight_alignment_to_four_bytes
  func @check_weight_alignment_to_four_bytes(%arg0: tensor<1x5x1x7x!quant.uniform<i8:f32, 0.0078160231932997704>>) -> tensor<1x49x!quant.uniform<i8:f32, 0.042479366064071655:2>> attributes {tf.entry_function = {inputs = "flatten_input_int8", outputs = "Identity_int8"}} {
// CHECK: tensor<49x36xi8>
    %0 = "tfl.pseudo_qconst"() {qtype = tensor<49x35x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, value = dense<1> : tensor<49x35xi8>} : () -> tensor<49x35x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>
    %1 = "tfl.pseudo_qconst"() {qtype = tensor<49x!quant.uniform<i32:f32, 6.1507766076829284E-5>>, value = dense<1> : tensor<49xi32>} : () -> tensor<49x!quant.uniform<i32:f32, 6.1507766076829284E-5>>
// CHECK: xc.fc
    %2 = "tfl.fully_connected"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x5x1x7x!quant.uniform<i8:f32, 0.0078160231932997704>>, tensor<49x35x!quant.uniform<i8<-127:127>:f32, 0.0078694447875022888>>, tensor<49x!quant.uniform<i32:f32, 6.1507766076829284E-5>>) -> tensor<1x49x!quant.uniform<i8:f32, 0.042479366064071655:2>>
// CHECK: return
    return %2 : tensor<1x49x!quant.uniform<i8:f32, 0.042479366064071655:2>>
}

// -----

// CHECK-LABEL: check_bias_with_per_channel_quantized_scale
  func @check_bias_with_per_channel_quantized_scale(%arg0: tensor<1x32x32x3x!quant.uniform<i8:f32, 0.0078431377187371254>>) -> tensor<1x10x!quant.uniform<i8:f32, 3.906250e-03:-128>> attributes {tf.entry_function = {inputs = "model/constant_fake_quantizer/PartitionedCall/Part", outputs = "model/softmax/Softmax"}} {
    %0 = "tfl.pseudo_const"() {value = dense<[[0, 0], [0, 0], [0, 0], [0, 1]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
    %1 = "tfl.pad"(%arg0, %0) : (tensor<1x32x32x3x!quant.uniform<i8:f32, 0.0078431377187371254>>, tensor<4x2xi32>) -> tensor<1x32x32x4x!quant.uniform<i8:f32, 0.0078431377187371254>>
    %2 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {0.003407094394788146,0.015011144801974297,0.0037102699279785156,0.020531326532363892,0.0084305340424180031,0.041204351931810379,0.047421257942914963,0.013000926002860069,0.010114379227161407,0.0081331897526979446,0.034491207450628281,0.019600516185164452,0.0077919322066009045,0.011329432018101215,0.009657515212893486,0.035633828490972519,0.013807233422994614,0.041861720383167267,0.015848748385906219,0.011077031493186951,0.024975625798106194,0.019187286496162415,0.018198961392045021,0.014922699891030788,0.010007184930145741,0.042646225541830063,0.00694607337936759,0.0048463833518326283,0.01082695834338665,0.013022688217461109,0.015287445858120918,0.044415559619665146}>>, value = dense<1> : tensor<32x3x3x4xi8>} : () -> tensor<32x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {0.003407094394788146,0.015011144801974297,0.0037102699279785156,0.020531326532363892,0.0084305340424180031,0.041204351931810379,0.047421257942914963,0.013000926002860069,0.010114379227161407,0.0081331897526979446,0.034491207450628281,0.019600516185164452,0.0077919322066009045,0.011329432018101215,0.009657515212893486,0.035633828490972519,0.013807233422994614,0.041861720383167267,0.015848748385906219,0.011077031493186951,0.024975625798106194,0.019187286496162415,0.018198961392045021,0.014922699891030788,0.010007184930145741,0.042646225541830063,0.00694607337936759,0.0048463833518326283,0.01082695834338665,0.013022688217461109,0.015287445858120918,0.044415559619665146}>>
    %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32:0, {2.6722309485194273E-5,1.177344674943015E-4,2.9100156098138541E-5,1.6103000962175429E-4,6.6121836425736547E-5,3.2317137811332941E-4,3.7193141179159284E-4,1.0196804214501753E-4,7.9328463471028954E-5,6.3789724663365632E-5,2.7051928918808699E-4,1.5372953203041106E-4,6.1113190895412117E-5,8.885829447535798E-5,7.5745221693068743E-5,2.7948099886998534E-4,1.0829202801687643E-4,3.2832723809406161E-4,1.2430391507223248E-4,8.687868103152141E-5,1.9588725990615785E-4,1.504885294707492E-4,1.4273695705924183E-4,1.1704077769536525E-4,7.8487726568710059E-5,3.3448019530624151E-4,5.4479005484608933E-5,3.8010850403225049E-5,8.4917322965338826E-5,1.0213873611064628E-4,1.1990153871010989E-4,3.4835733822546899E-4}>>, value = dense<[14060, 8605, 15519, -6578, 25659, -4847, 3786, 83, -12795, -2362, 5725, -6015, -485, 4946, 9048, 4572, 19154, 4846, 8063, 22876, 7174, 8506, -14636, -13299, -11893, 4758, 17466, 9777, -23668, 7782, 7757, 4389]> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32:0, {2.6722309485194273E-5,1.177344674943015E-4,2.9100156098138541E-5,1.6103000962175429E-4,6.6121836425736547E-5,3.2317137811332941E-4,3.7193141179159284E-4,1.0196804214501753E-4,7.9328463471028954E-5,6.3789724663365632E-5,2.7051928918808699E-4,1.5372953203041106E-4,6.1113190895412117E-5,8.885829447535798E-5,7.5745221693068743E-5,2.7948099886998534E-4,1.0829202801687643E-4,3.2832723809406161E-4,1.2430391507223248E-4,8.687868103152141E-5,1.9588725990615785E-4,1.504885294707492E-4,1.4273695705924183E-4,1.1704077769536525E-4,7.8487726568710059E-5,3.3448019530624151E-4,5.4479005484608933E-5,3.8010850403225049E-5,8.4917322965338826E-5,1.0213873611064628E-4,1.1990153871010989E-4,3.4835733822546899E-4}>>
    %4 = "tfl.conv_2d"(%1, %2, %3) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x32x32x4x!quant.uniform<i8:f32, 0.0078431377187371254>>, tensor<32x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {0.003407094394788146,0.015011144801974297,0.0037102699279785156,0.020531326532363892,0.0084305340424180031,0.041204351931810379,0.047421257942914963,0.013000926002860069,0.010114379227161407,0.0081331897526979446,0.034491207450628281,0.019600516185164452,0.0077919322066009045,0.011329432018101215,0.009657515212893486,0.035633828490972519,0.013807233422994614,0.041861720383167267,0.015848748385906219,0.011077031493186951,0.024975625798106194,0.019187286496162415,0.018198961392045021,0.014922699891030788,0.010007184930145741,0.042646225541830063,0.00694607337936759,0.0048463833518326283,0.01082695834338665,0.013022688217461109,0.015287445858120918,0.044415559619665146}>>, tensor<32x!quant.uniform<i32:f32:0, {2.6722309485194273E-5,1.177344674943015E-4,2.9100156098138541E-5,1.6103000962175429E-4,6.6121836425736547E-5,3.2317137811332941E-4,3.7193141179159284E-4,1.0196804214501753E-4,7.9328463471028954E-5,6.3789724663365632E-5,2.7051928918808699E-4,1.5372953203041106E-4,6.1113190895412117E-5,8.885829447535798E-5,7.5745221693068743E-5,2.7948099886998534E-4,1.0829202801687643E-4,3.2832723809406161E-4,1.2430391507223248E-4,8.687868103152141E-5,1.9588725990615785E-4,1.504885294707492E-4,1.4273695705924183E-4,1.1704077769536525E-4,7.8487726568710059E-5,3.3448019530624151E-4,5.4479005484608933E-5,3.8010850403225049E-5,8.4917322965338826E-5,1.0213873611064628E-4,1.1990153871010989E-4,3.4835733822546899E-4}>>) -> tensor<1x16x16x32x!quant.uniform<i8:f32, 0.023529412224888802>>
    %5 = "tfl.custom"(%4) {custom_code = "XC_bsign_8", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x16x16x32x!quant.uniform<i8:f32, 0.023529412224888802>>) -> tensor<1x16x16x1xi32>
    %6 = "tfl.pseudo_const"() {value = dense<1> : tensor<32x3x3x1xi32>} : () -> tensor<32x3x3x1xi32>
    %7 = "tfl.pseudo_const"() {value = dense<[129, 132, 125, 132, 127, 132, 131, 131, 119, 131, 132, 129, 101, 128, 131, 133, 124, 129, 130, 131, 121, 125, 133, 130, 130, 133, 131, 132, 131, 129, 130, 122]> : tensor<32xi32>} : () -> tensor<32xi32>
    %8 = "tfl.custom"(%5, %6, %7) {custom_code = "XC_bconv2d_bin", custom_option = opaque<"tfl", "0x696C6C6567616C5F706172616D730073747269646500020202040470616464696E670003240A17030103010015680428062401"> : tensor<51xi8>} : (tensor<1x16x16x1xi32>, tensor<32x3x3x1xi32>, tensor<32xi32>) -> tensor<1x8x8x1xi32>
    %9 = "tfl.pseudo_const"() {value = dense<1> : tensor<64x3x3x1xi32>} : () -> tensor<64x3x3x1xi32>
    %10 = "tfl.pseudo_const"() {value = dense<[0.0123519404, 0.0157591589, 0.00894330721, 0.0102242995, 0.0199145842, 0.00856406335, 0.0132913068, 0.0141302636, 0.0140401693, 0.00828326307, 0.0122382315, 0.00907240342, 0.0174852721, 0.0111954007, 0.00830175448, 0.0124316616, 0.0116535919, 0.0131482836, 0.0135210874, 0.0166070238, 0.0147064291, 0.013818399, 0.0105420081, 0.0169008784, 1.250870e-02, 0.0140332924, 0.0201447383, 0.0161800738, 0.0132619087, 0.0175907314, 0.0164049603, 0.0163851678, 0.00905151572, 0.0121403169, 0.0175230317, 0.00960024353, 0.0141542088, 0.0188811254, 0.0117896646, 0.0131512601, 0.0126701752, 0.0130300876, 0.0102129066, 0.0109999655, 0.0200717784, 0.0218085889, 0.0139973331, 0.0114720603, 0.0105320374, 0.0266707316, 0.00898698344, 0.0164819043, 0.0201003309, 0.0215392876, 0.0118977632, 0.0125502767, 0.0200995263, 0.0109535884, 0.0182847939, 0.00788395106, 0.0171663351, 0.015060599, 0.0139129283, 0.0174452141]> : tensor<64xf32>} : () -> tensor<64xf32>
    %11 = "tfl.pseudo_const"() {value = dense<[-0.0597040392, -0.0318945721, -0.109959833, -0.181978762, -0.0831351652, -0.140071258, -0.0647479519, -0.0554356165, -0.0421276502, -0.117031187, -0.0800563469, -0.0927631258, -0.0404938385, -0.0849730819, -0.109576233, -0.0600946546, -0.0810963734, -0.0664368569, -0.149310246, -0.0604892187, -0.0578607395, -7.586790e-02, -0.0872217044, -0.0533980168, -0.0586776063, -0.0976688489, -0.0608845279, -0.0805006474, -0.047402285, -0.0729817376, -0.0423240438, -0.0965159684, -0.0762302428, -0.0821381658, -0.0371011794, -0.0861711502, -0.0631853789, -0.021644786, -0.0850465968, -0.0621159896, -0.04860048, -0.0578661971, -0.0720161572, -0.0612344816, -0.0406481773, -0.0377969071, -0.0595206209, -0.0548714064, -0.0624929592, -0.0308750719, -0.0634970739, -0.0647348687, -0.0287438743, -0.0294320304, -0.0624713562, -0.078123182, -0.0510106906, -0.0831354856, -0.0362704918, -8.491640e-02, -0.0337719917, -0.092276059, -0.0381127596, -0.0648368895]> : tensor<64xf32>} : () -> tensor<64xf32>
    %cst = constant unit
    %12 = "tfl.custom"(%8, %9, %10, %11, %cst) {custom_code = "LceBconv2d", custom_option = opaque<"tfl", "0x6368616E6E656C735F696E0064696C6174696F6E5F6865696768745F666163746F720064696C6174696F6E5F77696474685F666163746F720066757365645F61637469766174696F6E5F66756E6374696F6E007061645F76616C7565730070616464696E67007374726964655F686569676874007374726964655F776964746800088277614C3329221508010820010101010002020404040404040404102401"> : tensor<160xi8>} : (tensor<1x8x8x1xi32>, tensor<64x3x3x1xi32>, tensor<64xf32>, tensor<64xf32>, none) -> tensor<1x4x4x64x!quant.uniform<i8:f32, 0.023529412224888802>>
    %13 = "tfl.pseudo_qconst"() {qtype = tensor<10x1024x!quant.uniform<i8<-127:127>:f32:0, {0.0079948892816901207,0.010309415869414806,0.0076371021568775177,0.0063417563214898109,0.0071352124214172363,0.0084017934277653694,0.0083739077672362328,0.0083578014746308327,0.0097445268183946609,0.0086075784638524055}>>, value = dense<1> : tensor<10x1024xi8>} : () -> tensor<10x1024x!quant.uniform<i8<-127:127>:f32:0, {0.0079948892816901207,0.010309415869414806,0.0076371021568775177,0.0063417563214898109,0.0071352124214172363,0.0084017934277653694,0.0083739077672362328,0.0083578014746308327,0.0097445268183946609,0.0086075784638524055}>>
// CHECK: "0x0000FFFF0000000000000000FFFF0000FFFFFFFF000000000000000000000000CF01BFFB7C035905590307004AFB8800B6F7B6FE0000000000000000000000000200020002000300020002000200020002000200000000000000000000000000544DB763DE49AD7A03454351FE50D650405E4153000000000000000000000000A805A805A805A805A805A805A805A805A805A805A805A805A805A805A805A80558FA58FA58FA58FA58FA58FA58FA58FA58FA58FA58FA58FA58FA58FA58FA58FA1500150015001500150015001500150015001500150015001500150015001500"
// CHECK-SAME: tensor<1x7x16xi16>
    %14 = "tfl.pseudo_qconst"() {qtype = tensor<10x!quant.uniform<i32:f32:0, {1.8811503832694143E-4,2.4257448967546225E-4,1.7969652253668755E-4,1.4921779802534729E-4,1.6788735229056329E-4,1.976892672246322E-4,1.9703313591890037E-4,1.9665416039060801E-4,2.2928298858460039E-4,2.0253124239388853E-4}>>, value = dense<[463, -1089, 892, 1369, 857, 7, -1206, 136, -2122, -330]> : tensor<10xi32>} : () -> tensor<10x!quant.uniform<i32:f32:0, {1.8811503832694143E-4,2.4257448967546225E-4,1.7969652253668755E-4,1.4921779802534729E-4,1.6788735229056329E-4,1.976892672246322E-4,1.9703313591890037E-4,1.9665416039060801E-4,2.2928298858460039E-4,2.0253124239388853E-4}>>
// CHECK: xc.fc
    %15 = "tfl.fully_connected"(%12, %13, %14) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x4x4x64x!quant.uniform<i8:f32, 0.023529412224888802>>, tensor<10x1024x!quant.uniform<i8<-127:127>:f32:0, {0.0079948892816901207,0.010309415869414806,0.0076371021568775177,0.0063417563214898109,0.0071352124214172363,0.0084017934277653694,0.0083739077672362328,0.0083578014746308327,0.0097445268183946609,0.0086075784638524055}>>, tensor<10x!quant.uniform<i32:f32:0, {1.8811503832694143E-4,2.4257448967546225E-4,1.7969652253668755E-4,1.4921779802534729E-4,1.6788735229056329E-4,1.976892672246322E-4,1.9703313591890037E-4,1.9665416039060801E-4,2.2928298858460039E-4,2.0253124239388853E-4}>>) -> tensor<1x10x!quant.uniform<i8:f32, 0.079715639352798462:-1>>
    %16 = "tfl.softmax"(%15) {beta = 1.000000e+00 : f32} : (tensor<1x10x!quant.uniform<i8:f32, 0.079715639352798462:-1>>) -> tensor<1x10x!quant.uniform<i8:f32, 3.906250e-03:-128>>
// CHECK: return
    return %16 : tensor<1x10x!quant.uniform<i8:f32, 3.906250e-03:-128>>
}