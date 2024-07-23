#!/bin/bash

# Define the parameter variations for each _AR parameter
ar_params=("_AR1_0" "_AR2_0" "_AR3_0" "_AR4_0" "_AR5_0" "_AR6_0")

# Define start and end time interval arrays for each file (assume they have pairs)
ar_start_params=(
"822601 955167 1495934 1732080 1761008 1796460 1820912 1834480 2563793 2721776 3009167 3217136 3498773 3736105 3778288 3856880 3865328 3922184 3936240 4093551 4099568 4642754 4725232 4909990 5001663 5033968 5054704 5742487 5754608 5788944 5870320 6055429 6083056 6143984 6148080 6191238 6221584 6236400 6531056"
"1697176 2428622 2693829 2820984 3060263 3176728 3497181 3594992 3775870 4010637 4065008 4261104 4617718 4670704 4729151 4859120 5063920 5325040 5501455 5923286 6084015 6243967 6282776 6484047 6776304"
"815622 823792 928991 1503390 1521648 1595888 1631248 1643248 1721341 1795622 1916400 1928176 1962736 2694886 2702832 3073198 3143664 3279374 3405181 3489520 4268853 4860656 4954864 4975088 5484927 5621744 5870975 5933552 6094064 6124528 6146032 6156528 6207472 6717424"
"546934 854086 917358 1495182 1811318 2070000 2085360 2179543 3481703 3757552 3891745 5349750 5571039 5717253 6102886 6182896 6187760"
"923224 1015280 1066480 1157360 1205488 1219312 1241584 1257712 1305840 1314800 1357040 1436400 1473520 1623792 1646576 1677808 1758704 3144846 3159536 3186928 3200599 3212016 3225840 3231728 3243504 3252720 3266032 3271152 3277552 3282672 3289328 3334128 3356656 3364080 3384048 3413744 3419888 3445488 3472791 3480048 3535856 3550960 3568624 3602045 3618800 3730285 3758064 3762928 3767280 3772912 3777264 3784432 3788784 3794928 3801328 3805680 3809776 3814128 3818992 3823088 3831536 3836400 3844080 4313840 4317936 4322544 4326896 4338928 4349936 4873200 4880368 4887536 4902384 4911600 4923888 4928240 4935152 4940784 4946672 4952560 4969712 4974064 4979952 4985584 4990448 5007344 5014000 5025264 5032688 5087366 5529254 5546992 5616880 5666544 5686256 5737712 5748208 5927920 5996272 6032656 6039280 6048752 6055664 6065392 6193271 6199792 6211824 6215920 6220784 6224880"
"1147460 1290264 1514224 1589365 1625584 1647600 1847222 2108023 2143087 2247015 2473712 2478064 2541552 2547696 2592520 2630384 2633200 2652656 2653936 2659568 2667760 2678000 2689776 2692592 2696944 2799344 2821872 2825968 2831856 2840048 2846704 2886384 2897904 2901488 3002608 3056431 3243207 3467335 3862256 3870192 3900144 3926000 3949040 3968752 4009456 4036607 4059120 4101104 4183792 4223984 4320006 4348952 4395159 4442096 4661744 4667120 4671728 4682224"
)

ar_end_params=(
"826352 1460208 1728240 1757168 1761040 1817072 1830640 1840912 2717936 2792157 3213296 3420709 3574000 3774448 3783696 3861488 3882776 3932400 3938832 4095728 4404976 4721392 4794096 4910576 5030128 5050864 5054992 5750768 5776144 5866480 6001936 6079216 6140144 6144240 6148462 6213648 6232560 6527216 6554608"
"2358256 2493936 2795248 3049712 3124496 3181336 3591152 3747312 3902349 4061168 4257264 4262645 4666864 4674207 4855280 5060080 5321200 5426590 5505520 5933080 6107120 6272280 6431728 6772464 6963480"
"819952 837616 990704 1517808 1592048 1614608 1639408 1648144 1747952 1912560 1924336 1958896 2684144 2698992 2707696 3139824 3144231 3280112 3485680 3506416 4856816 4951024 4971248 5004304 5617904 5784048 5929712 6090224 6120688 6142192 6152688 6203632 6713584 6734608"
"697584 878576 1377304 1651736 2066160 2081520 2089318 2712304 3753712 3757758 4533528 5359128 5580056 5958128 6179056 6183920 6188056"
"1011440 1062640 1153520 1201648 1215472 1237744 1253872 1302000 1310960 1353200 1432560 1469680 1619952 1642736 1673968 1754864 1853168 3155696 3183088 3191024 3208176 3222000 3227888 3239664 3248880 3262192 3267312 3273712 3278832 3285488 3330288 3352816 3360240 3380208 3409904 3416048 3441648 3447792 3476208 3532016 3547120 3564784 3569838 3614960 3628528 3738352 3759088 3763440 3769072 3773168 3780592 3784944 3791088 3797488 3801840 3805936 3810288 3815152 3819248 3827696 3832560 3840240 3845948 4314096 4318704 4323056 4335088 4346096 4350416 4876528 4883696 4898544 4907760 4920048 4924400 4931312 4936944 4942832 4948720 4965872 4970224 4976112 4981744 4986608 5003504 5010160 5021424 5028848 5034224 5092880 5543152 5613040 5662704 5682416 5733872 5744368 5924080 5946096 6017552 6035440 6044912 6051824 6061552 6067736 6195952 6207984 6212080 6216688 6221040 6234864"
"1243632 1510384 1545200 1621744 1643760 1648614 2012397 2109424 2180678 2469872 2474224 2540528 2547440 2557680 2629872 2632688 2648560 2652912 2654192 2663920 2676720 2678768 2691312 2693104 2704880 2818032 2822128 2828016 2836208 2844400 2847216 2896368 2899952 2902000 3003376 3172592 3387632 3858416 3866352 3896304 3922160 3945200 3964912 4005616 4015088 4055280 4097264 4179952 4220144 4238360 4341016 4362776 4438256 4452336 4662000 4667376 4681968 4774128"
)

# Loop through each file in ar_params
for (( i=0; i<${#ar_params[@]}; i++ )); do
  file="${ar_params[$i]}"
  
  start_line="${ar_start_params[$i]}"
  #echo "Start line: $start_line"
  end_line="${ar_end_params[$i]}"
  #echo "End line: $end_line"

  # Get number of elements using tr and wc -w
  num_elements=$(echo "$start_line" | tr ' ' '\n' | wc -w)
  
  # Loop through elements in the current line (using num_elements)
  for (( j=1; j<=num_elements ; j++ )); do
    start_val=$(echo "$start_line" | cut -d ' ' -f $j)
    end_val=$(echo "$end_line" | cut -d ' ' -f $j)

    # Build the command string with string interpolation
    command_str="python optimizer.py -f ${file} -d Recordings -start ${start_val} -end ${end_val} -dbd -p"
	
    echo "Launching command: $command_str"
	$command_str > /dev/null 2>&1 &

  done
  
  # Build the command string without start and end
  command_str="python optimizer.py -f ${file} -d Recordings -dbd -p"

  echo "Launching command: $command_str"
  $command_str > /dev/null 2>&1 &
  
done

