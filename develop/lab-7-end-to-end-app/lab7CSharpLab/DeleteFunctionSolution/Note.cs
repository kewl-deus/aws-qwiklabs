﻿using System;
using System.Collections.Generic;
using System.Text;
using Amazon.DynamoDBv2.DataModel;

namespace DeleteFunctionSolution
{
    [DynamoDBTable("pollynotes")]
    public class Note
    {
        [DynamoDBHashKey]
        public string userId { get; set; }

        [DynamoDBRangeKey]
        public string noteId { get; set; }

        [DynamoDBProperty]
        public string note { get; set; }
    }
}
